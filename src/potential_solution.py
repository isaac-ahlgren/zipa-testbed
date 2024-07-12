import os, pickle
from multiprocessing import Pool, Lock, synchronize, get_context, Process
import multiprocessing.queues
import sys
_is_windows= sys.platform == 'win32'
if _is_windows:
    import _winapi

def work(q, reply_q):
    print("Worker: Main says", q.get())
    reply_q.put('haha')


class DupSemLockHandle(object):
    """
    Picklable wrapper for a handle. Attempts to mirror how PipeConnection objects are pickled using appropriate api
    """

    def __init__(self, handle, pid=None):
        if pid is None:
            # We just duplicate the handle in the current process and
            # let the receiving process steal the handle.
            pid = os.getpid()
        if _is_windows:
            proc = _winapi.OpenProcess(_winapi.PROCESS_DUP_HANDLE, False, pid)
            try:
                self._handle = _winapi.DuplicateHandle(
                    _winapi.GetCurrentProcess(),
                    handle, proc, 0, False, _winapi.DUPLICATE_SAME_ACCESS)
            finally:
                _winapi.CloseHandle(proc)
        else:
            self._handle = handle
        self._pid = pid

    def detach(self):
        """
        Get the handle, typically from another process
        """
        # retrieve handle from process which currently owns it
        if self._pid == os.getpid():
            # The handle has already been duplicated for this process.
            return self._handle

        if not _is_windows:
            return self._handle

        # We must steal the handle from the process whose pid is self._pid.
        proc = _winapi.OpenProcess(_winapi.PROCESS_DUP_HANDLE, False,
                                   self._pid)
        try:
            return _winapi.DuplicateHandle(
                proc, self._handle, _winapi.GetCurrentProcess(),
                0, False, _winapi.DUPLICATE_CLOSE_SOURCE | _winapi.DUPLICATE_SAME_ACCESS)
        finally:
            _winapi.CloseHandle(proc)


def reduce_lock_connection(self):
    sl = self._semlock
    dh = DupSemLockHandle(sl.handle)
    return rebuild_lock_connection, (dh, type(self), (sl.kind, sl.maxvalue, sl.name))


def rebuild_lock_connection(dh, t, state):
    handle = dh.detach()  # Duplicated handle valid in current process's context

    # Create a new instance without calling __init__ because we'll supply the state ourselves
    lck = t.__new__(t)
    lck.__setstate__((handle,)+state)
    return lck


# Add our own reduce function to pickle SemLock and it's child classes
synchronize.SemLock.__reduce__ = reduce_lock_connection


class PicklableQueue(multiprocessing.queues.Queue):
    """
    A picklable Queue that skips the call to context.assert_spawning because it's no longer needed
    """

    def __init__(self, *args, **kwargs):
        ctx = get_context()
        super().__init__(*args, **kwargs, ctx=ctx)

    def __getstate__(self):

        return (self._ignore_epipe, self._maxsize, self._reader, self._writer,
                self._rlock, self._wlock, self._sem, self._opid)

def is_locked(l):
    """
    Returns whether the given lock is acquired or not.
    """
    locked = l.acquire(block=False)
    if locked is False:
        return True
    else:
        l.release()
        return False


if __name__ == '__main__':
    """
    from multiprocessing import set_start_method
    set_start_method('spawn')
    """

    print(f'The platform is {sys.platform}.')

    # Example that shows that you can now pickle/unpickle locks and they'll still point towards the same object
    l1 = Lock()
    p = pickle.dumps(l1)
    l2 = pickle.loads(p)
    print('before acquiring, l1 locked:', is_locked(l1), 'l2 locked', is_locked(l2))
    l2.acquire()
    print('after acquiring l1 locked:', is_locked(l1), 'l2 locked', is_locked(l2))

    # Putting a queue to a queue:
    q1 = PicklableQueue()
    q1.put('Putting a queue to a queue!')
    q2 = PicklableQueue()
    q2.put(q1)
    print(q2.get().get())

    # This should still work:
    q = PicklableQueue()
    reply_q = PicklableQueue()
    q.put('It still works!')
    p = Process(target=work, args=(q, reply_q))
    p.start()
    print(reply_q.get())
    p.join()

    # Example that shows that you can now pickle/unpickle locks and they'll still point towards the same object
    # Example that shows how you can pass a queue to Pool and it will work
    with Pool(1) as pool:
        q.put('laugh')
        pool.apply(work, (q, reply_q))
        print(reply_q.get())