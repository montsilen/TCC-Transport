# TCC-Transport

双车协商通行实验与实测轨迹处理代码。规则实验比较固定折扣与 TTC 自适应折扣；PPO 实验复用历史策略，支持实测初态和固定释放后的巡航恢复。

## 运行双车实验

将基线仓库 `montsilen/TCC-Driving` 检出到本仓库的同级目录。实验沿用其提交 `8e9891e20f68d563476c5e1dae40f4e42c14d954` 中的议价模块、车辆模型和 PPO 策略。使用装有 NumPy 和 PyTorch 的 Python 环境，在本仓库根目录执行以下 PowerShell 命令。

```powershell
$env:PYTHONPATH = (Resolve-Path ../TCC-Driving).Path
python run_two_car_discount_experiment.py --out Result/two_car_discount
python run_two_car_ppo_discount_experiment.py --help
```

PPO 复跑需通过 `--checkpoint` 指定本地完整权重。数据处理入口包括 INTERACTION 车对筛选、Waymo 提取、NGSIM、SinD 和 Songdo 轨迹标准化，以及场景合并和观测条件筛选。Waymo 提取另需 protobuf。运行各入口的 `--help` 可查看输入和输出参数。CARLA 的 Windows 环境配置见 [工具说明](tools/carla/README.md)。

原始轨迹、整理后的场景清单、权重和运行结果均在本地保存。
