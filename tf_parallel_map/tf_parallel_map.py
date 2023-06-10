import atexit
import ctypes
import itertools
import multiprocessing as mp
import os
import queue
import signal
import threading

import more_itertools
import numpy as np
import tensorflow as tf

_pool = None


def initialize_pool(n_workers=None, flag_namespace_getter=None):
    if n_workers is None:
        n_workers = min(len(os.sched_getaffinity(0)), 12)

    # important to use 'spawn', because 'fork' would mean the whole memory is (lazily) copied
    # then due to copy-on-write semantics, it gets duplicated when the parent changes anything
    ctx = mp.get_context('spawn')
    global _pool

    if flag_namespace_getter is None:
        _pool = ctx.Pool(n_workers, initializer=_init_worker_process)
    else:
        flag_values = flag_namespace_getter()
        _pool = ctx.Pool(
            n_workers, initializer=_init_worker_process_with_flags,
            initargs=(flag_values, flag_namespace_getter,))
    return _pool


def build_dataflow(
        examples, load_fn, extra_load_fn_args, learning_phase, batch_size, rng=None,
        n_completed_steps=0, n_total_steps=None, n_test_epochs=1, roundrobin_sizes=None):
    is_training = learning_phase.lower().startswith('train')
    if is_training:
        n_total_items = int(n_total_steps * batch_size if n_total_steps is not None else None)
    elif learning_phase.lower().startswith('val'):
        n_total_items = None
    else:
        n_total_items = int(len(examples) * n_test_epochs)

    ds = parallel_map_as_tf_dataset(
        load_fn, examples, shuffle_before_each_epoch=is_training,
        extra_args=extra_load_fn_args, rng=rng, max_unconsumed=batch_size * 2,
        n_completed_items=n_completed_steps * batch_size, n_total_items=n_total_items,
        roundrobin_sizes=roundrobin_sizes)

    return ds.batch(batch_size, drop_remainder=is_training)


def parallel_map_as_tf_dataset(
        fun, iterable, *, shuffle_before_each_epoch=False,
        extra_args=None, rng=None, max_unconsumed=256, n_completed_items=0,
        n_total_items=None, roundrobin_sizes=None, disable_parallelity=False):
    """Maps `fun` to each element of `iterable` and wraps the resulting sequence as
    as a TensorFlow Dataset. Elements are processed by parallel workers using `multiprocessing`.

    Args:
        fun: A function that takes an element from `iterable` plus `extra_args` and returns a
        sequence of
            numpy arrays.
        iterable: An iterable holding the inputs.
        shuffle_before_each_epoch: Shuffle the input elements before each epoch. Converts
            `iterable` to a list internally.
        extra_args: extra arguments in addition to an element from `iterable`,
        rng: A random number generator for shuffling. If `None`, a new one is created.
        max_unconsumed: Maximum number of unconsumed elements allowed to exist in the output
        queue before blocking.
        n_completed_items: Number of items that have already been processed. Useful for resuming
        training.
        n_total_items: Total number of items to process.
        roundrobin_sizes: If not `None`, then `iterable` is assumed to be a list of lists.
            Each list in `iterable` is iterated over in a round-robin fashion, with the given sizes.
            This is useful for training on multiple datasets simultaneously.
        disable_parallelity: If `True`, then no parallelity is used. This is useful for debugging.
    Returns:
        tf.data.Dataset based on the arrays returned by `fun`.
        """

    extra_args = extra_args or []

    # Automatically determine the output tensor types and shapes by calling the function on
    # the first element
    if roundrobin_sizes:
        (first_elem,), iterable[0] = more_itertools.spy(iterable[0])
    else:
        (first_elem,), iterable = more_itertools.spy(iterable)

    sample_output = fun(first_elem, *extra_args, rng=np.random.default_rng())
    output_signature = tf.nest.map_structure(tf.type_spec_from_value, sample_output)

    if roundrobin_sizes:
        items = roundrobin_iterate_repeatedly(
            iterable, roundrobin_sizes, shuffle_before_each_epoch, rng)
    else:
        items = iterate_repeatedly(
            iterable, shuffle_before_each_epoch, new_rng(rng))

    # If we are restoring from a checkpoint and have already completed some
    # training steps for that checkpoint, then we need to advance the RNG
    # accordingly, to continue exactly where we left off.
    iter_rng = new_rng(rng)
    advance_rng(iter_rng, n_completed_items)
    items = itertools.islice(items, n_completed_items, n_total_items)

    if disable_parallelity:
        def gen():
            for item in items:
                yield fun(item, *extra_args, new_rng(iter_rng))
    else:
        gen = parallel_map_as_generator(
            fun, items, extra_args, rng=iter_rng, max_unconsumed=max_unconsumed)

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    # Make the cardinality of the dataset known to TF.
    if n_total_items is not None:
        ds = ds.take(n_total_items - n_completed_items)
    return ds


def parallel_map_as_generator(fun, items, extra_args, max_unconsumed=256, rng=None):
    semaphore = threading.Semaphore(max_unconsumed)
    q = queue.Queue()
    end_of_sequence_marker = object()
    should_stop = False

    if _pool is None:
        raise RuntimeError("Pool not initialized. Call `initialize_pool` first.")

    def producer():
        for i_item, item in enumerate(items):
            if should_stop:
                break
            semaphore.acquire()
            q.put(_pool.apply_async(fun, (item, *extra_args, new_rng(rng))))

        q.put(end_of_sequence_marker)

    def consumer():
        while (future := q.get()) is not end_of_sequence_marker:
            value = future.get()
            semaphore.release()
            yield value

    def stop():
        nonlocal should_stop
        should_stop = True
        _pool.close()
        _pool.terminate()

    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()
    atexit.register(stop)

    return consumer


def _init_worker_process_with_flags(flag_values, flag_namespace_getter):
    flags_target = flag_namespace_getter()
    for key in flag_values.__dict__:
        setattr(flags_target, key, getattr(flag_values, key))
    _init_worker_process()


def _init_worker_process():
    _terminate_on_parent_death()
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _terminate_on_parent_death():
    prctl = ctypes.CDLL("libc.so.6").prctl
    PR_SET_PDEATHSIG = 1
    prctl(PR_SET_PDEATHSIG, signal.SIGTERM)


def iterate_repeatedly(seq, shuffle_before_each_epoch=False, rng=None):
    """Iterates over and yields the elements of `iterable` over and over.
    If `shuffle_before_each_epoch` is True, the elements are put in a list and shuffled before
    every pass over the data, including the first."""

    if rng is None:
        rng = np.random.default_rng()

    # create a (shallow) copy so shuffling only applies to the copy.
    seq = list(seq)
    rng.shuffle(seq)
    yield from seq

    while True:
        if shuffle_before_each_epoch:
            rng.shuffle(seq)
        yield from seq


def roundrobin_iterate_repeatedly(
        seqs, roundrobin_sizes, shuffle_before_each_epoch=False, rng=None):
    iters = [iterate_repeatedly(seq, shuffle_before_each_epoch, new_rng(rng)) for seq in seqs]
    return roundrobin(iters, roundrobin_sizes)


def roundrobin(iterables, sizes):
    iterators = [iter(iterable) for iterable in iterables]
    for iterator, size in zip(itertools.cycle(iterators), itertools.cycle(sizes)):
        for _ in range(size):
            try:
                yield next(iterator)
            except StopIteration:
                return


MAX_INT = 2 ** 32 - 1


def new_rng(rng):
    return np.random.Generator(np.random.PCG64(rng.integers(MAX_INT)))


def advance_rng(rng, num_generated_ints):
    for _ in num_generated_ints:
        rng.integers(MAX_INT)
