from pandas import DataFrame, DatetimeIndex, Series, Timestamp, DateOffset
import numpy as np
import feather
import numba
import pyximport


def build_offset_data(start=Timestamp('1970-01-01 00:00:00'), days=365*130):
    index = DatetimeIndex(start=start, freq='d', periods=days)
    offset_info = DataFrame({'base': index.asi8,
                             'decades': (index + DateOffset(years=10)).asi8,
                             'years': (index + DateOffset(years=1)).asi8,
                             'months': (index + DateOffset(months=1)).asi8})
    feather.write_dataframe(offset_info, path='./offset_data.feather')


class PrecomputedOffset:
    try:
        offsets = feather.read_dataframe('./offset_data.feather')
    except feather.FeatherError:
        offsets = None
    
    def __init__(self, years=0, months=0, days=0):
        self.decades = years // 10
        years_months = months // 12
        self.years = years - 10 * self.decades + years_months
        self.months = months - 12 * years_months
        self.days = days

        # index array is stored without datetime64 metadata
        self.index_arr = self.offsets.base.values
        self.decades_arr = self.offsets.decades.values
        self.years_arr = self.offsets.years.values
        self.months_arr = self.offsets.months.values
        
    def __add__(self, other):
        if isinstance(other, DatetimeIndex):
            new_calendar = self.add_decades(other.asi8)
            new_calendar = self.add_years(new_calendar)
            new_calendar = self.add_months(new_calendar)
            new_calendar = self.add_days(new_calendar)
            return DatetimeIndex(new_calendar)
        
        elif isinstance(other, Timestamp):
            new_date = self.add_decades(np.array([other.value]))
            new_date = self.add_years(new_date)
            new_date = self.add_months(new_date)
            new_date = self.add_days(new_date)

            return Timestamp(new_date[0])
        else:
            raise ValueError(
                "Supported types for '__add__' are DatetimeIndex and Timestamp"
            )

    def __radd__(self, other):
        return self.__add__(other)

    def add_decades(self, date):
        if self.decades == 0:
            return date
        
        idx = self.index_arr.searchsorted(date)
        step = self.decades_arr[idx]
    
        if self.decades == 1:
            return step
            
        for year in range(self.years-1):
            idx = self.index_arr.searchsorted(step)
            step = self.decades_arr[idx]
    
        return step      
        
    def add_years(self, date):
        if self.years == 0:
            return date
        
        idx = self.index_arr.searchsorted(date)
        step = self.years_arr[idx]
    
        if self.years == 1:
            return step
            
        for year in range(self.years-1):
            idx = self.index_arr.searchsorted(step)
            step = self.years_arr[idx]
    
        return step
    
    
    def add_months(self, date):
        if self.months == 0:
            return date
        
        idx = self.index_arr.searchsorted(date)
        step = self.months_arr[idx]
    
        if self.months == 1:
            return step
            
        for year in range(self.months-1):
            idx = self.index_arr.searchsorted(step)
            step = self.months_arr[idx]
    
        return step
    
    
    def add_days(self, date):
        idx = self.index_arr.searchsorted(date)
        return self.index_arr[idx + self.days]
    
