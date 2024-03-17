# Copyright (C) 2021 Akselos
import contextlib
import os
import tempfile
import uuid
import bz2
import gzip
import io


def decompress_data(data, filename):
    """Decompress the given data.  The filename is given only to determine the compression type."""
    if filename.endswith('.bz2'):
        filename = filename[:-4]
        data = bz2.decompress(data)
    elif filename.endswith('.gz'):
        filename = filename[:-3]
        data = gzip.GzipFile(fileobj=io.BytesIO(data)).read()

    return data, filename


@contextlib.contextmanager
def atomic_write(filepath, yield_filepath=False, mode='w'):
    """
    A context manager that yields a file that can be written to.  The writes to the file
    show up atomically in the given filepath when the context manager exits.  This avoids the
    unusual but potentially important case where there is an exception or other interruption
    (e.g. process is killed) while a file is being written, which will leave a partially written
    file on disk.  If such a file is read at a later time it may cause a crash.

    If the filepath already exists, it will be removed before it is replaced by the new file
    (this part isn't atomic-- there may be rare circumstances in which the old file is removed but
    the new one is never written.  If an exception occurs before the context manager exists,
    filepath remains unchanged.

    There is more subtle than it would seem at first:
    http://stackoverflow.com/questions/2333872/atomic-writing-to-file-with-python
    http://stackoverflow.com/questions/7799680/python-need-to-close-file-from-mkstemp-if-i-use-fdopen

    [#3414] Caveat on `tempfile.NamedTemporaryFile` (on Window only!).
      Quoted from python doc
        Whether the name can be used to open the file a second time, while the named
        temporary file is still open, varies across platforms (it can be so used on Unix;
        it cannot on Windows NT or later).

      This leads to several nasty side-effects on Windows.
      1) os.remove() silently fails, and the file is NOT removed.
      2) os.unlink() raises exception (PermissionError)
      3) os.rename() raises exception (FileExistsError)

    If `atomic_write` will be used to write a temporary file, one needs to use
    `tempfile.mkstemp` and close it immediately, or our `utils.reserve_tmp_filepath`.
    One can't use `tempfile.NamedTemporaryFile` even though it has a benefit
    of explicit scoping & cleanup afterward.
    Thus, in both cases, manual cleanup is required.
    This caveat doesn't exists on unix.

    """
    # Create the temp file in the same directory as the target file so rename will work.
    # We don't use mkstemp here to avoid Python issue 1473760, which can effectively cause a hang.
    filepath = os.path.abspath(filepath)
    target_dir, filename = os.path.split(filepath)
    temp_filepath = os.path.join(target_dir, uuid.uuid4().hex + '-' + filename)
    file = open(temp_filepath, mode=mode)
    try:
        if yield_filepath:
            yield temp_filepath
        else:
            yield file
    except Exception as e:
        print(e)
        file.close()
        os.remove(temp_filepath)
        raise

    file.flush()
    os.fsync(file.fileno())
    file.close()

    if os.path.isfile(filepath):
        try:
            os.remove(filepath)
        except Exception as e:
            # Eh, not sure why I get this sometimes.  The file is in use somewhere, but I think
            # it's been closed already...  see if this "fixes" the problem.
            print(e)

    os.rename(temp_filepath, filepath)


def sanitize_filename(unsafe_filename):
    accepted_characters = ('.', '_', ' ')
    safe_filename = "".join(
        c for c in unsafe_filename if c.isalnum() or c in accepted_characters).rstrip()
    safe_filename = safe_filename.replace(' ', '_')
    return safe_filename
