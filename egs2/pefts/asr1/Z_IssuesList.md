1. Maxlenratio issue. sometimes, large maxlenratio might cause the following error. All you need to do is to reduce the maxlenratio in conf/decoding/**.yaml to proper value, e.g., 0.3 to 0.25.
```
2025-03-28 03:52:18,698 (asr_inference:522) INFO: speech length: 507200
2025-03-28 03:52:18,712 (beam_search:428) INFO: decoder input length: 1500
2025-03-28 03:52:18,712 (beam_search:429) INFO: max output length: 450
2025-03-28 03:52:18,713 (beam_search:430) INFO: min output length: 0
Traceback (most recent call last):
File "/root/espnet/tools/miniconda/envs/espnet/lib/python3.8/runpy.py", line 194, in _run_module_as_main
    return _run_code(code, main_globals, None,
File "/root/espnet/tools/miniconda/envs/espnet/lib/python3.8/runpy.py", line 87, in _run_code
    exec(code, run_globals)
File "/root/espnet/espnet2/bin/asr_inference.py", line 1184, in <module>
    main()
File "/root/espnet/espnet2/bin/asr_inference.py", line 1180, in main
    inference(**kwargs)
File "/root/espnet/espnet2/bin/asr_inference.py", line 853, in inference
    results = speech2text(**batch)
File "/root/espnet/tools/miniconda/envs/espnet/lib/python3.8/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
File "/root/espnet/espnet2/bin/asr_inference.py", line 559, in __call__
    results = self._decode_single_sample(enc[0])
File "/root/espnet/espnet2/bin/asr_inference.py", line 652, in _decode_single_sample
    nbest_hyps = self.beam_search(
File "/root/espnet/tools/miniconda/envs/espnet/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
File "/root/espnet/tools/miniconda/envs/espnet/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
File "/root/espnet/espnet/nets/beam_search.py", line 437, in forward
    best = self.search(running_hyps, x, pre_x=pre_x)
File "/root/espnet/espnet/nets/batch_beam_search.py", line 291, in search
    scores, states = self.score_full(
File "/root/espnet/espnet/nets/batch_beam_search.py", line 194, in score_full
    scores[k], states[k] = d.batch_score(hyp.yseq, hyp.states[k], x)
File "/root/espnet/espnet2/asr/decoder/whisper_decoder.py", line 225, in batch_score
    logp, states = self.forward_one_step(ys, torch.empty(0), xs, cache=None)
File "/root/espnet/espnet2/asr/decoder/whisper_decoder.py", line 180, in forward_one_step
    self.decoders.token_embedding(tgt)
RuntimeError: The size of tensor a (449) must match the size of tensor b (448) at non-singleton dimension 1
```
2. inference_nj does not work. No matter how big it is, there is only one nj at a time. This is because the run.pl sets the max_jobs_run with inference_nj. We fix this issue by simply adding --max-jobs-run "${_nj}" to the 1614-th line of asr.sh as follows:
```bash
${_cmd} --gpu "${_ngpu}" --max-jobs-run "${_nj}" JOB=1:"${_nj}" "${_logdir}"/asr_inference.JOB.log \
```
3. There is no free space left. 这是因为inodes满了，不是真正没有存储空间了，解决方法如下：
```bash
df -i 
# 查看哪个文件夹导致inodes满了（100%）
# cd到某个目录下，使用 find . -printf "%i\n" | sort -u | wc -l 查看这个目录占了多少inodes
# 一般就是把不需要的文件删掉即可，我这里是把一些corpora的最原始的数据删除了
```
4. 租AutoDL算力时，如果发现VSCODE不能提交远程，先运行以下代码即可
```bash
source /etc/network_turbo
```
5. 在跑CDSD-partB第一阶段时，发现kaldi格式的数据量（约为4w条）与原始数据（约为9.2w条）不一致，检查后发现*/10h/Text/中“20-label.txt”不对，需改为“20_label.txt”，这样改完就好了。
6. fairseq installation following "./installers/install_fairseq.sh" causes bugs when nets training. We solve this by simply replacing the "./tools/fairseq" file with the correspoding ones that download from official fairseq. However, another BUG appears: the token ID seems only cover 1 and 8. This leads to failure of training. We find out that in the run.sh, we apply --cleaner whisper_basic for both SSL models and whisper models.
Note that the whisper_basic is specifical for whisper models. For SSL models, we set cleaner to "none". BUG solved!
7. the func "make_pad_mask" from espnet2.asr.encoder.hubert_encoder causes CUDA out of memory. We replace it with self.easy_make_pad_mask() which is writed by myself.
8. Error while loading conda entry point.
```
conda-libmamba-solver (libarchive.so.20: cannot open shared object file: No such file or directory)  

CondaValueError: You have chosen a non-default solver backend (libmamba) but it was not recognized. Choose one of: classic
```
```
The following instructions might solve this issue:
1. conda activate your-conda-environment-name
2. uninstall libarchive, conda-libmamba-solver, and libmamba
3. open /root/.condarc, and comment out the original channels and add the new channels as follows:
    channels:
    - https://software.repos.intel.com/python/conda/
    - conda-forge
    # - https://software.repos.intel.com/python/conda/
    # - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
    # - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
    # - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
    # - defaults
    show_channel_urls: true
4. conda install libarchive, conda-libmamba-solver, and libmamba
*Note that this solution might cause other issues
```