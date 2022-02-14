# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

# --- Do not remove these libs ---
from functools import reduce
from typing import Any, Callable, Dict, List

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from skopt.space import Categorical, Dimension, Integer, Real  # noqa

from freqtrade.optimize.hyperopt_interface import IHyperOpt

# --------------------------------
# Add your lib to import here
import talib.abstract as ta  # noqa
import freqtrade.vendor.qtpylib.indicators as qtpylib


class TemaHyperopt(IHyperOpt):
    """
    This is a Hyperopt template to get you started.

    More information in the documentation: https://www.freqtrade.io/en/latest/hyperopt/

    You should:
    - Add any lib you need to build your hyperopt.

    You must keep:
    - The prototypes for the methods: populate_indicators, indicator_space, buy_strategy_generator.

    The methods roi_space, generate_roi_table and stoploss_space are not required
    and are provided by default.
    However, you may override them if you need 'roi' and 'stoploss' spaces that
    differ from the defaults offered by Freqtrade.
    Sample implementation of these methods will be copied to `user_data/hyperopts` when
    creating the user-data directory using `freqtrade create-userdir --userdir user_data`,
    or is available online under the following URL:
    https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/templates/sample_hyperopt_advanced.py.
    """
    @staticmethod
    def populate_indicators(dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        This method can also be loaded from the strategy, if it doesn't exist in the hyperopt class.
        """
        dataframe['CMO'] = ta.CMO(dataframe, timeperiod = 25)
        dataframe['RSI'] = ta.RSI(dataframe, timeperiod = 25)
        dataframe['TEMA'] = ta.TEMA(dataframe, timeperiod = 50)
      
        # Bollinger bands
        bollingerTA1 = ta.BBANDS(dataframe, timeperiod=25, nbdevup=1.0, nbdevdn=1.0, matype=0)
        
        dataframe['bb_lowerbandTA1'] = bollingerTA1['lowerband']
        dataframe['bb_middlebandTA1'] = bollingerTA1['middleband']
        dataframe['bb_upperbandTA1'] = bollingerTA1['upperband']
        
        bollingerTA2 = ta.BBANDS(dataframe, timeperiod=25, nbdevup=2.0, nbdevdn=2.0, matype=0)
        
        dataframe['bb_lowerbandTA2'] = bollingerTA2['lowerband']
        dataframe['bb_middlebandTA2'] = bollingerTA2['middleband']
        dataframe['bb_upperbandTA2'] = bollingerTA2['upperband']
        
        bollingerTA3 = ta.BBANDS(dataframe, timeperiod=25, nbdevup=3.0, nbdevdn=3.0, matype=0)
        
        dataframe['bb_lowerbandTA3'] = bollingerTA3['lowerband']
        dataframe['bb_middlebandTA3'] = bollingerTA3['middleband']
        dataframe['bb_upperbandTA3'] = bollingerTA3['upperband']
        
        bollingerTA4 = ta.BBANDS(dataframe, timeperiod=25, nbdevup=4.0, nbdevdn=4.0, matype=0)
        
        dataframe['bb_lowerbandTA4'] = bollingerTA4['lowerband']
        dataframe['bb_middlebandTA4'] = bollingerTA4['middleband']
        dataframe['bb_upperbandTA4'] = bollingerTA4['upperband']
        
        return dataframe

    @staticmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:
        """
        Define the buy strategy parameters to be used by Hyperopt.
        """
        def populate_buy_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:
            """
            Buy strategy Hyperopt will build and use.
            """
            conditions = []

            # GUARDS AND TRENDS
            if params.get('cmo-enabled'):
                conditions.append(dataframe['CMO'] < params['cmo-value'])
            if params.get('rsi-enabled'):
                conditions.append(dataframe['RSI'] < params['rsi-value'])

            # TRIGGERS
            if 'trigger' in params:
                if params['trigger'] == 'bb_lower':
                    conditions.append(dataframe['close'] < dataframe['bb_lowerbandTA1'])
                    
                if params['trigger'] == 'bb_tema_middle':
                    conditions.append(dataframe['TEMA'] < dataframe['bb_middlebandTA1'])
                if params['trigger'] == 'bb_cross_middle2':
                    conditions.append(qtpylib.crossed_above(
                        dataframe['TEMA'], dataframe['bb_middlebandTA2']
                    ))
                if params['trigger'] == 'bb_cross_lower2':
                    conditions.append(qtpylib.crossed_above(
                        dataframe['TEMA'], dataframe['bb_lowerbandTA2']
                    ))
                if params['trigger'] == 'bb_cross_lower1':
                    conditions.append(qtpylib.crossed_above(
                        dataframe['TEMA'], dataframe['bb_lowerbandTA1']
                    ))
                    
                    

            # Check that the candle had volume
            conditions.append(dataframe['volume'] > 0)

            if conditions:
                dataframe.loc[
                    reduce(lambda x, y: x & y, conditions),
                    'buy'] = 1

            return dataframe

        return populate_buy_trend

    @staticmethod
    def indicator_space() -> List[Dimension]:
        """
        Define your Hyperopt space for searching buy strategy parameters.
        """
        return [
            Integer(-35, 35, name='cmo-value'),
            Integer(5, 70, name='rsi-value'),
            Categorical([True, False], name='cmo-enabled'),
            Categorical([True, False], name='rsi-enabled'),
            Categorical(['bb_lower', 'bb_tema_middle', 'bb_cross_middle2',"bb_cross_lower2","bb_cross_lower1"], name='trigger')
        ]

    @staticmethod
    def sell_strategy_generator(params: Dict[str, Any]) -> Callable:
        """
        Define the sell strategy parameters to be used by Hyperopt.
        """
        def populate_sell_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:
            """
            Sell strategy Hyperopt will build and use.
            """
            conditions = []

            # GUARDS AND TRENDS
            if params.get('sell-rsi-enabled'):
                conditions.append(dataframe['RSI'] > params['sell-rsi-value'])
            if params.get('sell-cmo-enabled'):
                conditions.append(dataframe['CMO'] > params['sell-cmo-value'])

            # TRIGGERS
            if 'sell-trigger' in params:
                if params['sell-trigger'] == 'sell-tema-lower':
                    conditions.append(dataframe['TEMA'] > dataframe['bb_lowerbandTA1'])
                if params['sell-trigger'] == 'sell-tema-middle':
                    conditions.append(dataframe['TEMA'] > dataframe['bb_middlebandTA1'])
                if params['sell-trigger'] == 'sell-tema-uper':
                    conditions.append(dataframe['TEMA'] > dataframe['bb_upperbandTA2'])
                if params['sell-trigger'] == 'sell-close-upper3':
                    conditions.append(dataframe['close'] > dataframe['bb_upperbandTA3'])
                if params['sell-trigger'] == 'sell-close-upper4':
                    conditions.append(dataframe['close'] > dataframe['bb_upperbandTA4'])

            # Check that the candle had volume
            conditions.append(dataframe['volume'] > 0)

            if conditions:
                dataframe.loc[
                    reduce(lambda x, y: x & y, conditions),
                    'sell'] = 1

            return dataframe

        return populate_sell_trend

    @staticmethod
    def sell_indicator_space() -> List[Dimension]:
        """
        Define your Hyperopt space for searching sell strategy parameters.
        """
        return [
            Integer(50, 100, name='sell-rsi-value'),
            Integer(0, 50, name='sell-cmo-value'),
            Categorical([True, False], name='sell-rsi-enabled'),
            Categorical([True, False], name='sell-cmo-enabled'),
            Categorical(['sell-tema-lower',
                         'sell-tema-middle',
                         'sell-tema-uper',
                         'sell-close-upper3',
                         'sell-close-upper4'], name='sell-trigger')
        ]

    @staticmethod
    def generate_roi_table(params: Dict) -> Dict[int, float]:
        """
        Generate the ROI table that will be used by Hyperopt
        This implementation generates the default legacy Freqtrade ROI tables.
        Change it if you need different number of steps in the generated
        ROI tables or other structure of the ROI tables.
        Please keep it aligned with parameters in the 'roi' optimization
        hyperspace defined by the roi_space method.
        """
        roi_table = {}
        roi_table[0] = params['roi_p1'] + params['roi_p2'] + params['roi_p3']
        roi_table[params['roi_t3']] = params['roi_p1'] + params['roi_p2']
        roi_table[params['roi_t3'] + params['roi_t2']] = params['roi_p1']
        roi_table[params['roi_t3'] + params['roi_t2'] + params['roi_t1']] = 0

        return roi_table

    @staticmethod
    def roi_space() -> List[Dimension]:
        """
        Values to search for each ROI steps
        Override it if you need some different ranges for the parameters in the
        'roi' optimization hyperspace.
        Please keep it aligned with the implementation of the
        generate_roi_table method.
        """
        return [
            Integer(10, 1200, name='roi_t1'),
            Integer(10, 800, name='roi_t2'),
            Integer(10, 600, name='roi_t3'),
            Real(0.01, 0.13, name='roi_p1'),
            Real(0.01, 0.15, name='roi_p2'),
            Real(0.01, 0.30, name='roi_p3'),
        ]
    
    
    @staticmethod
    def stoploss_space() -> List[Dimension]:
        """
        Stoploss Value to search
        Override it if you need some different range for the parameter in the
        'stoploss' optimization hyperspace.
        """
        return [
            Real(-0.15, -0.05, name='stoploss'),
        ]
    
    
    @staticmethod
    def trailing_space() -> List[Dimension]:
        """
        Create a trailing stoploss space.
        You may override it in your custom Hyperopt class.
        """
        return [
            # It was decided to always set trailing_stop is to True if the 'trailing' hyperspace
            # is used. Otherwise hyperopt will vary other parameters that won't have effect if
            # trailing_stop is set False.
            # This parameter is included into the hyperspace dimensions rather than assigning
            # it explicitly in the code in order to have it printed in the results along with
            # other 'trailing' hyperspace parameters.
            Categorical([True], name='trailing_stop'),

            Real(0.01, 0.35, name='trailing_stop_positive'),

            # 'trailing_stop_positive_offset' should be greater than 'trailing_stop_positive',
            # so this intermediate parameter is used as the value of the difference between
            # them. The value of the 'trailing_stop_positive_offset' is constructed in the
            # generate_trailing_params() method.
            # This is similar to the hyperspace dimensions used for constructing the ROI tables.
            Real(0.001, 0.3, name='trailing_stop_positive_offset_p1'),

            Categorical([True, False], name='trailing_only_offset_is_reached'),
        ]

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators.
        Can be a copy of the corresponding method from the strategy,
        or will be loaded from the strategy.
        Must align to populate_indicators used (either from this File, or from the strategy)
        Only used when --spaces does not include buy
        """
        dataframe.loc[
            (
            # (qtpylib.crossed_below(dataframe["TEMA"], dataframe["bb_lowerbandTA"]))
                  
            (qtpylib.crossed_above(dataframe["TEMA"], dataframe["bb_middlebandTA1"]))
            & 
              (dataframe['CMO']>0) 
                  
                
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators.
        Can be a copy of the corresponding method from the strategy,
        or will be loaded from the strategy.
        Must align to populate_indicators used (either from this File, or from the strategy)
        Only used when --spaces does not include sell
        """
        dataframe.loc[
            (
                

            ((qtpylib.crossed_below(dataframe["CMO"],-25)) 
              & (dataframe["TEMA"]>=dataframe["bb_lowerbandTA1"])) 
            |
            ((qtpylib.crossed_above(dataframe["CMO"],25)) 
              & (dataframe["TEMA"]<=dataframe["bb_middlebandTA1"]))
                
                
            ),
            'sell'] = 1  
        
        return dataframe