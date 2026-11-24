# TCC-Transport

双车协商通行实验与实测轨迹处理代码。规则实验比较固定折扣与 TTC 自适应折扣；PPO 实验复用历史策略，支持实测初态和固定释放后的巡航恢复。

## 运行双车实验

| 目录 | 用途 |
| --- | --- |
| `traffic_data/sources/` | 各数据源的轨迹读取、格式转换与车对提取 |
| `traffic_data/` | 双车场景清单的构建、补全、合并与筛选 |
| `test/` | 双车折扣对照、PPO 复跑和 CARLA 冒烟测试 |
| `tools/carla/` | CARLA 环境安装与诊断 |

将基线仓库 `montsilen/TCC-Driving` 检出到本仓库的同级目录。实验沿用其提交 `8e9891e20f68d563476c5e1dae40f4e42c14d954` 中的议价模块、车辆模型和 PPO 策略。使用装有 NumPy 和 PyTorch 的 Python 环境，在本仓库根目录执行以下 PowerShell 命令。

```powershell
$env:PYTHONPATH = (Resolve-Path ., ../TCC-Driving).Path -join [IO.Path]::PathSeparator
python test/run_two_car_discount_experiment.py --out Result/two_car_discount
python test/run_two_car_ppo_discount_experiment.py --help
```

PPO 复跑需通过 `--checkpoint` 指定本地完整权重。CARLA 的 Windows 环境配置见 [工具说明](tools/carla/README.md)。

## 运行轨迹处理工具

设置上述 `PYTHONPATH` 后，在仓库根目录用 `python -m` 运行模块。以下命令分别查看 SinD 轨迹标准化和双车场景构建的输入输出参数。Waymo 提取另需 protobuf。

```powershell
python -m traffic_data.sources.normalize_sind_tracks --help
python -m traffic_data.build_real_traffic_cases --help
```

原始轨迹、整理后的场景清单、权重和运行结果均在本地保存。
