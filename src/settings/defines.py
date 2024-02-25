from __future__ import annotations

import json
import os
from typing import List

from pydantic import BaseModel

from settings import settings


class Defines(BaseModel):
    unchangeable_features: List[str] = [
        'срсод2', 'диастолическое артериальное давление при поступлении',
        'частота сердечных сокращений при поступлении', 'рост', 'АСТ',
        'ширина распределения эритроцитов (RDW-SD))', 'эритроц2', 'сробем2',
        'базофилы', 'билирубин', 'средний объем эритроцитов',
        'миелоциты', 'На', 'Локализация',
        'систолическое артериальное давление при поступлении', 'глю', 'пневмофиброз', 'р2св2',
        'палочкоядерные нейтрофилы', 'хроническая болезнь почек', 'возраст', 'МН(клапаны)', 'срконц2',
        'сробьем2', 'Отеки_при пост',
        'курение', 'средний объем тромбоцитов', 'Атеросклероз сосудов', 'полихимиотерапия', 'анемия', 'Активность',
        'Вид рака', 'ПИКС в анамнезе',
        'Лучевая терапия', 'пол', 'застой', 'Q', 'водно-электр нарушения', 'структурность',
        'хроническая язвенная болезнь', 'нейродегенеративные нарушения', 'очаг',
        'инф', 'Кардиомегалия', 'сахарный диабет', 'аневризма аорты', 'Операция', 'метастазы', 'ST', 'гидроторакс',
        'сосуд.рис', 'гипотиреоз', 'дислипидемия',
        'артериальная гипертензия', ' хронический бронхит', 'гепатоз', 'ожирение', 'потеря веса', 'ПИКС количество',
        'МА', 'острая сердечная недостаточность по Killip', 'ТЭЛА',
    ]
    features = [
        'ПТИ',
        'ТЭЛА',
        'креатинин',
        'МНО',
        'моноциты',
        'острая сердечная недостаточность по Killip',
        # 'ПТВ',
        'частота сердечных сокращений при поступлении',
        # 'гемоглобин',
        # 'глю',
        # 'сегментоядерные нейтрофилы',
        # 'ширина распределения тромбоцитов',
        # 'систолическое артериальное давление при поступлении',
        # 'гематокрит',
        # 'лимфоциты',
        # 'среднее содержание гемоглобина в эритроците',
        # 'ширина распределения эритроцитов (RDW-CV))',
        # 'средний объем эритроцитов',
        # 'NYHA',
        # 'вес',
        # 'количество эритроцитов',
        # 'эозинофилы',
        # 'тропонин',
        # 'АЛТ',
        # 'содержание крупных тромбоцитов',
        # 'кфк',
        # 'тромбоциты',
        # 'тромбокрит',
        # 'диастолическое артериальное давление при поступлении',
        # 'общий холестерин',
        # 'Отеки_при пост',
        # 'Локализация',
        # 'Вид рака',
        # 'лейкоциты',
        # 'скорость оседания эритроцитов',
        # 'хроническая язвенная болезнь',
        # 'средняя концентрация эритроцитов',
        # 'билирубин',
        # 'На',
        # 'средний объем тромбоцитов'
    ]
    features_hard_to_change = (
        "моноциты", "лимфоциты", "эозинофилы", "МНО", "ширина распределения эритроцитов (RDW-CV))",
        "ширина распределения тромбоцитов", "тромбокрит", "средняя концентрация эритроцитов", "тропонин",
        "содержание крупных тромбоцитов", "среднее содержание гемоглобина в эритроците"
    )

    changeable_features = []
    feature_limits = {
        "": [1, 2]
    }

    recommended_limits = {
        "": [1, 2]
    }

    HARD_TO_CHANGE_K = 10

    feature_change_coef = dict()

    def save_defines(self):
        os.makedirs(os.path.dirname(settings.defines_path), exist_ok=True)

        with open(settings.defines_path, "w") as f:
            json.dump(self.dict(), f, ensure_ascii=False, sort_keys=False, indent=4)
            # pickle.dump(self, f)

    @staticmethod
    def create_defines(save_when_init: bool = True) -> Defines:
        if os.path.exists(settings.defines_path):
            with open(settings.defines_path, "r") as f:
                # return pickle.load(f)
                return Defines(**json.load(f))
        else:
            res = Defines.__init_defines(Defines())

            if save_when_init:
                res.save_defines()
            return res

    @staticmethod
    def __init_defines(_defines: Defines) -> Defines:
        res = Defines()
        res.features = sorted(res.features, key=lambda x: 0 if x in res.unchangeable_features else 1, reverse=True)
        res.changeable_features = [f for f in res.features if f not in res.unchangeable_features]
        res.feature_change_coef = {
            f: res.HARD_TO_CHANGE_K if f in res.features_hard_to_change else 1
            for f in res.features if f not in res.unchangeable_features
        }

        return res

    def set_attr(self, attr_name: str, value: str):
        setattr(self, attr_name, value)
        self.save_defines()


defines = Defines.create_defines()
