# 清单目录

`libero_episode_index.json` 只保存公开数据的轻量 episode→task 索引；`train_*.json` 保存
按协议生成的训练/验证 episode ID。视频、图像、权重和日志不放在 Git 中。

`libero_data_files.json` 是同一 revision 的真实 parquet shard 清单。由于当前 Hub 数据
重新分片，不能直接使用 `meta/episodes` 的 `data/file_index` 拼路径；该清单由
`sim-vla-build-data-manifest` 读取远程 Parquet 页脚生成，`sim-vla-prepare-data` 按它
选择性下载并校验 episode 覆盖。

`eval_initializations.json` 是仿真评测的初始化序列，不是演示 episode 清单：固定
`batch_size=1` 时，官方 LeRobot evaluator 的 `episode_index=0..9` 为开发切片，
`10..29` 为最终切片。正式运行保存完整 30 episode 记录，再按该边界汇总。

若索引尚未下载，使用 `sim-vla-make-manifests --episode-index <meta/episodes.parquet>`
生成清单；脚本会拒绝缺少三个选定任务或存在跨任务重复的输入。

来源仓库、固定 revision、元信息快照和本机下载大小记录在
`manifests/source_provenance.json`；资产/权重本体仍由 `.gitignore` 排除。
