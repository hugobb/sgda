from pathlib import Path
from datetime import datetime
import uuid
import json
from typing import TypedDict, Optional
import tempfile
from omegaconf import OmegaConf
import pickle


class RecordInfo(TypedDict):
    id: str
    name: str
    date: str
    path: str


class Record:
    def __init__(self, info: Optional[RecordInfo] = None) -> None:
        if info is None:
            self.fp = tempfile.TemporaryDirectory()
            info = self.createInfo(Path(self.fp.name))

        self.id = info["id"]
        self.name = info["name"]
        self.date = info["date"]
        self.path = Path(info["path"])

        self.config = self.load_config()
        self.metrics = self.load_metrics()

    def save_config(self, config):
        self.config = config
        with open(self.path / "config.yaml", "wb") as fp:
            pickle.dump(config, fp)

    def load_config(self):
        filename = self.path / "config.yaml"
        if filename.is_file():
            with open(filename, 'rb') as fp:
                return pickle.load(fp)
        else:
            return None

    def load_metrics(self):
        filename = self.path / "metrics.yaml"
        if filename.is_file():
            return OmegaConf.load(filename)
        else:
            return None

    def save_metrics(self, metrics):
        self.metrics = OmegaConf.structured(dict(metrics))
        OmegaConf.save(self.metrics, self.path / "metrics.yaml")

    @staticmethod
    def createInfo(path: Path, name= ""):
        _id = str(uuid.uuid4())
        path = path / _id
        info = {"id": _id, "name": name, "date": datetime.now().isoformat(), "path": str(path)}
        path.mkdir(parents=True)
        return info


class ExpInfo(TypedDict):
    id: str
    name: str
    date: str
    path: str
        

class Experiment:
    def __init__(self, info: ExpInfo) -> None:
        self.id = info["id"]
        self.name = info["name"]
        self.date = info["date"]
        self.path = Path(info["path"])

        self.records = {}
        self.load_records()

    def __getitem__(self, key: str) -> Record:
        return self.records[key]

    def load_records(self) -> None:
        list_exp = self.path.glob("*/.info.json")
        for path in list_exp:
            with open(path, "r") as fp:
                info = json.load(fp)
                record = Record(info)
                self.records[record.id] = record

    def create_record(self, name: str = "") -> Record:
        info = Record.createInfo(self.path, name)
        with open(Path(info["path"])/ ".info.json", "w") as fp:
            json.dump(info, fp)
        
        record = Record(info)
        self.records[record.id] = record
        print("Record: %s" % record.id)
        return record

    def refresh(self):
        self.load_records()


class Database:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.experiments = {}
        self.loadExp()
    
    def __getitem__(self, key: str) -> Experiment:
        return self.experiments[key]

    def getRecord(self, key: str) -> Record:
        exp_id = list(self.log_dir.glob("*/%s" % key))[0].parts[-2]
        return self.experiments[exp_id][key]

    def refresh(self):
        self.loadExp()

    def loadExp(self) -> None:
        list_exp = self.log_dir.glob("*/.info.json")
        for path in list_exp:
            with open(path, "r") as fp:
                info = json.load(fp)
                exp = Experiment(info)
                self.experiments[exp.id] = exp

    def createExp(self, name: str = "") -> Experiment:
        _id = str(uuid.uuid4())
        path = self.log_dir / _id
        filename =  path / ".info.json"
        info = {"id": _id, "name": name, "date": datetime.now().isoformat(), "path": str(path)}
        
        path.mkdir(parents=True)
        with open(filename, "w") as fp:
            json.dump(info, fp)
        
        exp = Experiment(info)
        self.experiments[exp.id] = exp
        print("Experiment: %s" % exp.id)
        return exp

    
