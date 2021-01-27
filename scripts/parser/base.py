from abc import ABC

from .default_methods.parse_person import parse_person_info
from .default_methods.parse_bank_details import parse_bank_details
from .default_methods.parse_telephone_number import parse_telephone_numbers


class BaseCheque(ABC):
    @classmethod
    def type_number(cls):
        return cls.TYPE_NUMBER

    @classmethod
    def type_name(cls):
        return cls.TYPE_NAME

    @staticmethod
    def parse_bank_details(image):
        return parse_bank_details(image)

    @staticmethod
    def parse_telephone_numbers(img):
        return parse_telephone_numbers(img)

    @staticmethod
    def parse_person_info(img):
        return parse_person_info(img)

    @classmethod
    def parse(cls, gray_img):
        bank_data = cls.parse_bank_details(gray_img)
        numbers = cls.parse_telephone_numbers(gray_img)
        person_data = cls.parse_person_info(gray_img)

        return {
            'persons': person_data,
            **numbers,
            **bank_data,
            'type_number': cls.type_number(),
            'type_name': cls.type_name()
        }
