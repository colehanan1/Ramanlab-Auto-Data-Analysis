"""One detector process at a time, hub-wide (mount check + trial traces)."""

import threading

GPU = threading.Semaphore(1)
