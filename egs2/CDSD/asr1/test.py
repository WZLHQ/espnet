'''
ERROR: Unexpected segmentation fault encountered in worker.
Traceback (most recent call last):
  File "/media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/tools/miniconda/envs/espnet/lib/python3.11/site-packages/torch/utils/data/dataloader.py", line 1310, in _try_get_data
    data = self._data_queue.get(timeout=timeout)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/tools/miniconda/envs/espnet/lib/python3.11/queue.py", line 180, in get
    self.not_empty.wait(remaining)
  File "/media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/tools/miniconda/envs/espnet/lib/python3.11/threading.py", line 331, in wait
    gotit = waiter.acquire(True, timeout)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/tools/miniconda/envs/espnet/lib/python3.11/site-packages/torch/utils/data/_utils/signal_handling.py", line 73, in handler
    _error_if_any_worker_fails()
RuntimeError: DataLoader worker (pid 293412) is killed by signal: Segmentation fault. 

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/CDSD/asr1/espnet2/bin/asr_train.py", line 23, in <module>
    main()
  File "/media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/CDSD/asr1/espnet2/bin/asr_train.py", line 19, in main
    ASRTask.main(cmd=cmd)
  File "/media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/CDSD/asr1/espnet2/tasks/abs_task.py", line 1225, in main
    cls.main_worker(args)
  File "/media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/CDSD/asr1/espnet2/tasks/abs_task.py", line 1593, in main_worker
    cls.trainer.run(
  File "/media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/CDSD/asr1/espnet2/train/trainer.py", line 343, in run
    all_steps_are_invalid = cls.train_one_epoch(
                            ^^^^^^^^^^^^^^^^^^^^
  File "/media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/CDSD/asr1/espnet2/train/trainer.py", line 605, in train_one_epoch
    for iiter, (utt_id, batch) in enumerate(
  File "/media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/CDSD/asr1/espnet2/train/reporter.py", line 266, in measure_iter_time
    retval = next(iterator)
             ^^^^^^^^^^^^^^
  File "/media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/tools/miniconda/envs/espnet/lib/python3.11/site-packages/torch/utils/data/dataloader.py", line 741, in __next__
    data = self._next_data()
           ^^^^^^^^^^^^^^^^^
  File "/media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/tools/miniconda/envs/espnet/lib/python3.11/site-packages/torch/utils/data/dataloader.py", line 1524, in _next_data
    idx, data = self._get_data()
                ^^^^^^^^^^^^^^^^
  File "/media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/tools/miniconda/envs/espnet/lib/python3.11/site-packages/torch/utils/data/dataloader.py", line 1473, in _get_data
    success, data = self._try_get_data()
                    ^^^^^^^^^^^^^^^^^^^^
  File "/media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/tools/miniconda/envs/espnet/lib/python3.11/site-packages/torch/utils/data/dataloader.py", line 1323, in _try_get_data
    raise RuntimeError(
RuntimeError: DataLoader worker (pid(s) 293412) exited unexpectedly
# Accounting: time=14870 threads=1
# Ended (code 1) at Mon Jul 13 19:58:07 CST 2026, elapsed time 14870 seconds

'''