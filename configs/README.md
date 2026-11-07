# Current scientific inputs

日常调参入口：[配置修改指南](../docs/configuration-guide.md)。YAML 按用途分段，
常用参数靠前；注释说明单位、预算优先级和算法适用范围。

Author Scenario, Algorithm and Experiment. Scenarios contain geometry, jobs and
fault rules; standalone workload experiment inputs are retired. Internal workload
models still generate jobs and are not a second authoring interface.

The four packaged maps in `src/smartsom/config/maps/` serve Studio and installed
packages. The two `scenarios/machine-breakdown/` scenes serve
the single [ISL experiment family](../experiments/implicit-sl/README.md)
(Small/Medium × Base/ISL/E1a-current PPO/Rule). Fixed test seeds and world hashes
are embedded in each Experiment; no separate authored Case files are needed.
Other small current examples
exercise CLI workflows; they are engineering fixtures, not historical campaigns.

Algorithms select native PPO/DDQN, Base/SL, Rule, Random, Morris or Justin.
All policy definitions explicitly name observation components. Custom rules and
observations use the same source-pinned Python factory mechanism.

Historical configs, independent workload YAML and old experiment directories are
in the external source archive, not maintained in this tree. Existing generated
runs and checkpoints remain in place and require their matching source.
