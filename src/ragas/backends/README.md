# Backend Architecture Guide

Simple plugin architecture for data storage backends. Implement one abstract class, register via entry points.

## Architecture

```
Registry (dict-like) → Backend (implements BaseBackend) → Storage
```

**Key Files:**
- `base.py` - Abstract interface (6 methods)
- `registry.py` - Plugin discovery & dict-like access
- `local_csv.py`, `local_jsonl.py` - Reference implementations

## Quick Start

**1. Implement BaseBackend:**
```python
from ragas.backends.base import BaseBackend


class MyBackend(BaseBackend):
    def __init__(self, connection_string: str):
        self.conn = connection_string

    def load_dataset(self, name: str) -> List[Dict[str, Any]]:
        # Load dataset from your storage
        return [{"id": 1, "text": "example"}]

    def save_dataset(
        self, name: str, data: List[Dict], model: Optional[Type[BaseModel]]
    ):
        # Save dataset to your storage
        pass

    # ... implement other 4 methods (see base.py)
```

**2. Register via entry points:**
```toml
# pyproject.toml
[project.entry-points."ragas.backends"]
"my_backend" = "my_package.backend:MyBackend"
```

**3. Use:**
```python
from ragas.backends import get_registry

registry = get_registry()
backend = registry["my_backend"](connection_string="...")
```

## Required Methods

**BaseBackend (6 methods):**
```python
# Data loading
def load_dataset(name: str) -> List[Dict[str, Any]]
def load_experiment(name: str) -> List[Dict[str, Any]]

# Data saving  
def save_dataset(name: str, data: List[Dict], model: Optional[Type[BaseModel]])
def save_experiment(name: str, data: List[Dict], model: Optional[Type[BaseModel]])

# Listing
def list_datasets() -> List[str]
def list_experiments() -> List[str]
```

## Registry Usage

**Dict-like interface:**
```python
from ragas.backends import get_registry

registry = get_registry()
print(registry)  # {'local/csv': <class 'LocalCSVBackend'>, ...}

# Access backend classes
backend_class = registry["local/csv"]
backend = backend_class(root_dir="./data")

# Check availability
if "my_backend" in registry:
    backend = registry["my_backend"]()
```

## Reference Implementations

**LocalCSVBackend** (`local_csv.py`):
- **Pattern:** File-based storage with CSV format
- **Init:** `LocalCSVBackend(root_dir="./data")`
- **Storage:** `{root_dir}/datasets/{name}.csv`, `{root_dir}/experiments/{name}.csv`
- **Features:** Directory auto-creation, UTF-8 encoding, proper CSV escaping

**LocalJSONLBackend** (`local_jsonl.py`):
- **Pattern:** File-based storage with JSONL format  
- **Init:** `LocalJSONLBackend(root_dir="./data")`
- **Storage:** `{root_dir}/datasets/{name}.jsonl`, `{root_dir}/experiments/{name}.jsonl`
- **Features:** Handles complex nested data, preserves types

**GDriveBackend** (`gdrive_backend.py`, see `gdrive_backend.md`):
- **Pattern:** Cloud storage with Google Sheets format
- **Init:** `GDriveBackend(folder_id, service_account_file)`
- **Storage:** Google Drive folder with sheets for datasets/experiments
- **Features:** Collaborative editing, cloud sync, multiple auth methods

## Implementation Patterns

**Common backend structure:**
```python
class MyBackend(BaseBackend):
    def __init__(self, **config):
        # Initialize connection/client
        
    def _get_storage_path(self, data_type: str, name: str):
        # Generate storage location
        
    def _load(self, data_type: str, name: str):
        # Generic load implementation
        
    def _save(self, data_type: str, name: str, data, model):
        # Generic save implementation
        
    # Implement required methods using _load/_save
    def load_dataset(self, name): return self._load("datasets", name)
    def save_dataset(self, name, data, model): self._save("datasets", name, data, model)
    # ... etc
```

**Error handling:**
```python
def load_dataset(self, name: str):
    try:
        return self._load("datasets", name)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset '{name}' not found")
    except ConnectionError:
        raise RuntimeError(f"Storage connection failed")
```

**Pydantic model handling:**
```python
def save_dataset(self, name: str, data: List[Dict], model: Optional[Type[BaseModel]]):
    if model:
        # Validate data against model if provided
        validated_data = [model(**item).model_dump() for item in data]
        self._save(name, validated_data)
    else:
        self._save(name, data)
```

## Testing Your Backend

```python
def test_backend():
    backend = MyBackend(config="test")
    
    # Test save/load cycle
    test_data = [{"id": 1, "text": "test"}]
    backend.save_dataset("test_dataset", test_data, None)
    loaded = backend.load_dataset("test_dataset")
    assert loaded == test_data
    
    # Test listing
    datasets = backend.list_datasets()
    assert "test_dataset" in datasets
```

## Plugin Development

**Full plugin structure:**
```
my-backend-plugin/
├── pyproject.toml              # Entry point configuration
├── src/my_backend/
│   ├── __init__.py            # Export backend class
│   └── backend.py             # Backend implementation
└── tests/
    └── test_backend.py        # Integration tests
```

**Entry point registration:**
```toml
[project.entry-points."ragas.backends"]
"s3" = "my_backend.backend:S3Backend"
"postgres" = "my_backend.backend:PostgresBackend"
```

**Install & use:**
```bash
pip install my-backend-plugin
python -c "from ragas.backends import get_registry; print(get_registry())"
```

## Registry Internals

**Discovery process:**
1. Registry loads entry points from group `"ragas.backends"`  
2. Each entry point maps `name -> backend_class`
3. Lazy loading - backends loaded on first access
4. Dict-like interface for easy access

**Debugging:**
```python
from ragas.backends import get_registry

registry = get_registry()

# Check what's available
print(f"Available backends: {list(registry.keys())}")

# Get backend info
for name in registry:
    backend_class = registry[name]
    print(f"{name}: {backend_class.__module__}.{backend_class.__name__}")
```

## Design Decisions

**Why BaseBackend instead of separate Project/DataTable backends?**
- Simpler: One interface to implement vs. two
- Clearer: Backend owns both storage and operations
- Flexible: Backends can optimize cross-operation concerns

**Why entry points vs. manual registration?**
- Extensible: Third-party backends without code changes
- Standard: Follows Python packaging conventions  
- Discoverable: Automatic registration on install

**Why dict-like registry?**
- Intuitive: Familiar `registry["name"]` access pattern
- Debuggable: Shows available backends in repr
- Flexible: Supports `in`, `keys()`, iteration

---

**Quick Start:** Copy `local_csv.py`, replace CSV logic with your storage, add entry point, done.