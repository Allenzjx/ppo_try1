# Blocks18–19 CSV：主控检查

独立 CSV 仅包含已完成 block18 的 512 条和 block19 的 2048 条，共 2560 条 / 263 列，global 155009–157568。现有构建器正常 exit0，未修改构建器或生产代码，未导出 XLSX。

主控读取全部 receipt 并独立 Import-Csv 核对：全部 credited/optimized/saved=true、teacher=false；8 条真实 terminal，其余 2552 条不据此虚构为完成回合。阶段计数为 P01 15、P02 685、P03 29、P04 8、P05 793、P06 150、P07 1、P08 1、P09 366、P10 3、P11 17、P12 492；P13 0。512+2048=2560，不重复旧 CSV。

来源仅为两份完成的策略 audit，分别读取 50,056,873 和 157,447,313 bytes；未读取教师前缀、正在进行的视频评估或大型 physical 流。scalarGridVerified/engineRecalculated 为 true，源行与 authored grid 全量逐值检查通过；已查看非截断 inspections/error-scan 回执。

主控实际查看 contact_columns.png：标题、8 个接触字段、布尔值与数据行均可读，长标识符按既有列宽换行，没有遮挡或裁切；不为 CSV 增加装饰图。缺测 body roll/pitch、线速度向量保留空白，实际关节来源的 margin fallback 仍明确区分于直接物理测量；不填零或伪造同 tick join。

这张表不是自然 P01 评估成功、RR 新放置、配对稳定性改善或新物理测试的证明。当前 video8 数据不在此表。

