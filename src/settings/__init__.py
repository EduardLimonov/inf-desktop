class Settings:
    fit_model: bool = False
    dataset_path = "resource/dataset/data.xlsx"

    model_path = "resource/model_dumps/model.PICKLE"
    train_size = 0.75

    test_recom_path = "resource/test_dumps/test_recom.csv"
    recalc_test: bool = False

    defines_path = "resource/defines/defines.json"
    core_manager_path = "resource/core_manager/core_manager.json"


settings = Settings()
