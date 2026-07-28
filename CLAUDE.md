# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Install

```bash
# Editable install with all extras (recommended for development)
uv pip install -e .[all]

# Minimal install
uv pip install py-data-juicer
```

Build system is Hatchling (`pyproject.toml`). A custom `hatch_build.py` compiles C++ extensions (minhash via pybind11, tokenize via Cython) for wheels.

## Commands

```bash
# Entry points
dj-process --config path/to/recipe.yaml   # Run data processing pipeline
dj-analyze --config path/to/recipe.yaml   # Run data analysis

# Tests — use the custom runner, not bare pytest
python tests/run.py --tag standalone --mode regression   # All local tests
python tests/run.py --tag ray --mode regression          # All Ray/distributed tests
python tests/run.py --tag standalone --mode partial      # Only tests for changed files

# Single test file with pytest directly
python -m pytest tests/ops/filter/test_text_length_filter.py -v

# Linting
pre-commit run --all-files
```

## Code Style

- **Black**: line-length 120, target py310
- **isort**: profile "black"
- **Flake8**: max-line-length 120; ignores E203, E501, BLK100, F541; `__init__.py` files ignore F401
- Pre-commit hooks also run detect-secrets and build-op-doc

## Architecture

### Operator Type Hierarchy

All operators inherit from `OP` (in `data_juicer/ops/base_op.py`):

| Type | Method to implement | Behavior |
|------|-------------------|----------|
| **Mapper** | `process_single(sample)` or `process_batched(samples)` | Transforms each sample; must NOT override `process()` |
| **Filter** | `compute_stats_single(sample, context)` + `process_single(sample)` | Two-phase: compute stats (map), then filter; must NOT override `compute_stats()` or `process()` |
| **Deduplicator** | `compute_hash(sample)` + `process(dataset)` | Hash then deduplicate |
| **Selector** | `process(dataset)` | Dataset-level selection |
| **Grouper** | `process(dataset)` | Groups samples into batches |
| **Aggregator** | `process_single(sample)` | Reduces grouped samples |

The `__init_subclass__` hook enforces that subclasses don't override the wrong methods (raises `TypeError` at class definition time).

### Operator Registration

Operators register via `@OPERATORS.register_module("op_name")` decorator (`data_juicer/utils/registry.py`). All built-in operators auto-register on import because each `ops/<category>/__init__.py` imports all concrete modules.

### Pipeline Execution Flow

1. `init_configs(args)` parses YAML, loads custom operators, builds argument parser with only the referenced operators
2. `ExecutorFactory.create(cfg)` creates the executor (`DefaultExecutor`, `RayExecutor`, or `PartitionedRayExecutor`)
3. `DefaultExecutor.run()`:
   - Loads dataset via `DatasetBuilder` → format-specific `LocalFormatter` → HuggingFace `Dataset` → `NestedDataset`
   - Instantiates operators via `load_ops()`: looks up `OPERATORS.modules[name](**kwargs)` for each entry in `process:` list
   - Optionally fuses compatible filters (`fuse_operators`)
   - Runs `dataset.process(ops)` which calls `op.run(dataset)` for each operator sequentially
   - Exports via `Exporter`

### Key Data Columns

Defined in `data_juicer/utils/constant.py`:

- `Fields.stats` (`__dj__stats__`) — scalar statistics written by Filters (used by `dj-analyze`)
- `Fields.meta` (`__dj__meta__`) — structured metadata/tags written by operators
- `Fields.context` (`__dj__context__`) — temporary inter-op communication (not persisted)

Filter stats should be **scalar** (bool/float/int) for compatibility with `Analyzer` (pandas `describe`). Structured data belongs in `Fields.meta`.

### Config System

YAML recipes specify `process:` as a list of `{operator_name: {kwargs}}`. The config system (`data_juicer/config/config.py`) uses `jsonargparse` and auto-extracts constructor parameters from operator type hints via `add_class_arguments`. Global keys like `text_key`, `image_key` are propagated to all operators via `update_op_attr()`.

### Custom Operators

Two mechanisms, both using `@OPERATORS.register_module()`:

1. **`custom_operator_paths` in YAML** — `load_custom_operators()` dynamically imports files (via `importlib.util.spec_from_file_location`) or packages (via `importlib.import_module` after adding parent dir to `sys.path`). Package directories must contain `__init__.py`.

2. **`data_juicer/_au/` extension package** — local extensions mirroring the `ops/` structure. The `_au/__init__.py` must explicitly import submodules to trigger registration. Not auto-imported by `data_juicer`; loaded via `custom_operator_paths: ['data_juicer/_au']` in YAML.

### Entry Point Wiring

`dj-process` and `dj-analyze` entry points map to `data_juicer.tools.process_data:main` etc., but the actual source files live in the top-level `tools/` directory. A custom `MetaPathFinder` in `data_juicer/tools/__init__.py` redirects imports for editable installs.

### Test Infrastructure

Tests use `unittest` with `DataJuicerTestCaseBase` (`data_juicer/utils/unittest_utils.py`), which provides `generate_dataset()` and handles standalone vs. Ray mode. Tests are tagged with `@TEST_TAG("standalone")` or `@TEST_TAG("ray")`. The custom runner `tests/run.py` supports partial mode (only tests matching changed files).

## Extension Tests (`tests_au/`)

Tests for `_au` operators live in `tests_au/`, mirroring the `tests/` structure. Acceptance scripts use the `accept_` prefix and include a `.sh` wrapper.

# 撰写文档规范

* 图表用mermaid, 数学相关的用LaTex, 必要时可以用py脚本画一些更能帮助读者理解的图片(图片中的文字用英文).
* 分析要深入仔细, 既要包括纵向分析(算法或方法的由来与演进历史, 以及在该算法或方法的基础上又演进和优化出了些什么解决类似问题的方法, 新老方法各有什么优缺点, 各适合应用到什么场景), 纵向分析(同时期同类算法的对比分析, 不同算法或方法各有什么优缺点, 各适合应用到什么场景), 和 消融分析(算法或方法中哪些点是在benchmark实验或实践中被证明有效的, 哪些点相对来说更有效, 哪些没那么有效).
* 系统或程序的设计要包括静态架构(组件图,类图,组件和类的职责与关系等等)和动态架构(数据流图,序列图,工作流图,不同场景下的各组件或类的调用与协调图.如果是算法还会涉及forward阶段的数据流,模型组件间的调用,以及backwawrd阶段的数据流,gradient流,哪些权重冻结哪些会被更新,和模型组件间的调用等等).
* 解释要深入浅出, 图文并茂, 可以举一些易于理解的例子帮助说明, 对关键的逻辑也要进行深入的代码解读, 要用严谨的科普论文的风格.

# 编码规范

* 尽量在`.vscode/settings.json`中指定的虚拟环境或指定的python所属的虚拟环境中开发或测试,实在不行才用系统级的python环境.
* 尽量利用`data_juicer`已有的功能与模块.
* 要遵守"扩展大于修改的原则".
* `b/d/` 放各个论文的分析, 各个算子的设计文档(含测试用例设计).
* `b/d/urdf/` 放设计,开发,测试等阶段所需要的urdf文件.
* `b/dsAnalyz/` 放一些对数据集的分析设计与报告.
* `b/scripts/` 放一些工具脚本.
* 新加的功能和代码要放在 `data_juicer/_au/` 里, 该文件夹中的目录要参考 `data_juicer/` 的目录设定, 所有定制化的扩展代码都写在 `data_juicer/_au/` 中的相应目录里.
* 测试用例和验收脚本要放在 `tests_au/` 里, 该文件夹中的目录要参考  `tests/` 的目录设定, 后续所有定制化的扩展代码的测试代码都写在 `tests_au/` 中的相应目录里.
* 验收脚本也写在 `tests_au/` 中的相应目录里, 验收脚本必须以`accept_`开头, 并且要有调用的sh脚本. 
* `demos_au/` 放一些在特定场景下可用的pipeline和脚本,可作为例子或开箱即用.
* 测试和验证数据集用更真实的如下数据集或用户指定的其它数据集:
   + `/mnt/r/DATA/tst/Galaxea-Open-World-Dataset/Connect_Router_Cables_20250625_002/`
* 配置文件yaml中引用定制化的扩展功能的方法不能用简单的单文件方式, 而是更企业化的模块引入或包引入方式.