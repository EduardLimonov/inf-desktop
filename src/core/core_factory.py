from core.base_core import CoreInterface
from core.core import Core
from core.core_manager import CoreManager


class CoreFactory:
    @staticmethod
    def create_core(create_core_manager: bool = True) -> CoreInterface:
        if create_core_manager:
            return CoreManager.init_from_factory()
        else:
            return Core()
