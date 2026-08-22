from pathlib import Path
NUM_TRAIN_DAYS = 11
_current_dir = Path(__file__).parent.resolve()

PROJECT_ROOT_DIR = _current_dir.parent.resolve()
CEBRA_DIR = (PROJECT_ROOT_DIR / "CEBRA-main").resolve()
CEBRA_Parallel_DIR = (PROJECT_ROOT_DIR / "CEBRA_Parallel").resolve()
DATA_DIR = (PROJECT_ROOT_DIR / "data").resolve()
DATASET_DIR = rf'E:\My repo\llm\CORP_data_release'
# DATASET_DIR = "/data/hossein/mm_project/CORP_data_release"
DATA_DIR = r'/data/hossein/mm_project/competitionData'
OUT_DATA_PATH = r'speech/speech_data.npz'
WHOLE_DATA_PATH = r'speech/speech_24.npz'