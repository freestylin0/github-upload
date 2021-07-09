import os
import sys
import time
import numpy as np
import pandas as pd



def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def process_header(header_bytes):
    header_data = {
        'active_mass_g': int.from_bytes(header_bytes[152:156], byteorder='little') / 1000000,
        'comment': header_bytes[2317:2414].decode('utf-8').strip('\00'),
        'creator': header_bytes[2167:2225].decode('utf-8').strip('\00'),
        'barcode': header_bytes[2433:2454].decode('utf-8').strip('\00'),
        'pn': 1,
        'step_file': header_bytes[2533:2592].decode('utf-8').strip('\00'),
        'model': 1,
        'current_limit': abs(int.from_bytes(header_bytes[2074:2078], byteorder='little', signed=True)),
        'machine_id': int.from_bytes(header_bytes[2090:2091], byteorder='little'),
        'row_id': int.from_bytes(header_bytes[2091:2092], byteorder='little'),
        'channel_id': int.from_bytes(header_bytes[2092:2093], byteorder='little'),
    }
    return header_data


def process_body_bytes(body_bytes, debug=False):
    body_dtype = np.dtype([
        ('aux_indicator', '<i2'),
        ('record_raw', '<i4'),
        ('cycle_raw', '<i4'),
        ('step_method', '<i2'),
        ('step_name_raw', '<i1'),
        ('step_raw', '<i1'),
        ('step_time', '<i8'),
        ('voltage', '<i4'),
        ('current', '<i4'),
        ('column_1', '<i4'),
        ('temp', '<i4'),
        ('capacity_chg', '<i8'),
        ('capacity_dchg', '<i8'),
        ('energy_chg', '<i8'),
        ('energy_dchg', '<i8'),
        ('year', '<i2'),
        ('month', '<i1'),
        ('day', '<i1'),
        ('hour', '<i1'),
        ('minute', '<i1'),
        ('second', '<i2'),
        ('current_range', '<i4'),
        ('column_2', '<i4'),
    ])

    np_data = np.frombuffer(body_bytes, body_dtype)
    return np_data


def process_body_np(body_np, current_limit, debug=False):
    aux_indicator = body_np[:]['aux_indicator']
    record_raw = body_np[:]['record_raw']
    cycle_raw = body_np[:]['cycle_raw']
    step_method = body_np[:]['step_method']
    step_name_raw = body_np[:]['step_name_raw']
    step_raw = body_np[:]['step_raw']
    step_time = body_np[:]['step_time']
    voltage = body_np[:]['voltage']
    current = body_np[:]['current']
    column_1 = body_np[:]['column_1']
    temp = body_np[:]['temp']
    capacity_chg = body_np[:]['capacity_chg']
    capacity_dchg = body_np[:]['capacity_dchg']
    energy_chg = body_np[:]['energy_chg']
    energy_dchg = body_np[:]['energy_dchg']
    year = body_np[:]['year']
    month = body_np[:]['month']
    day = body_np[:]['day']
    hour = body_np[:]['hour']
    minute = body_np[:]['minute']
    second = body_np[:]['second']
    current_range = body_np[:]['current_range']
    column_2 = body_np[:]['column_2']

    # convert columns from integers into actual units
    step_time_s = step_time/1000
    voltage_V = voltage / 10000
    temp_C = temp / 10

    offsets_dict = {
        # something weird happens if I make the offset too large. Just for 10, I'll divide by 1000 again after I convert it to a dataframe
        10: [[current_range == 1, current_range == 10],
             [10_000, 1_000]],
        6000: [[current_range == 0, current_range == 100, current_range == 6000],
               [1_000_000, 100_000, 10_000]],
        50000: [[current_range == 0, current_range == 50000],
                [10_000, 10_000]],
        100000: [[abs(current_range) == 0, abs(current_range) == 10000, abs(current_range) == 50000, abs(current_range) == 100000],
                 [100_000, 100_000, 100_000, 100_000]],
    }
    offset_lists = offsets_dict[current_limit]

    offset = np.select(offset_lists[0], offset_lists[1], 1)

    current_A = current / offset
    capacity_chg_Ah = capacity_chg / (offset * 3600)
    capacity_dchg_Ah = capacity_dchg / (offset * 3600)
    energy_chg_Wh = energy_chg / (offset * 3600)
    energy_dchg_Wh = energy_dchg / (offset * 3600)

    # Combine converted arrays and split main data and auxt data
    result = np.vstack((
        aux_indicator,
        record_raw,
        cycle_raw,
        step_method,
        step_name_raw,
        step_raw,
        step_time_s,
        voltage_V,
        current_range,
        offset,
        current_A,
        temp_C,
        capacity_chg_Ah,
        capacity_dchg_Ah,
        energy_chg_Wh,
        energy_dchg_Wh,
        year,
        month,
        day,
        hour,
        minute,
        second
    )).T

    return result


def process_body_df(body_np, current_limit, debug=False):
    body_df = pd.DataFrame(body_np, columns=[
                'aux_indicator',
                'record_raw',
                'cycle_raw',
                'step_method',
                'step_name_raw',
                'step_raw',
                'step_time_s',
                'voltage_V',
                'current_range',
                'offset',
                'current_A',
                'temp_C',
                'capacity_chg_Ah',
                'capacity_dchg_Ah',
                'energy_chg_Wh',
                'energy_dchg_Wh',
                'year',
                'month',
                'day',
                'hour',
                'minute',
                'second'
    ])

    # divide some columns again because I couldn't do it in process_body_np without bugs.
    if current_limit == 10:
        body_df['current_A'] = body_df['current_A'] / 1000
        body_df['capacity_chg_Ah'] = body_df['capacity_chg_Ah'] / 1000
        body_df['capacity_dchg_Ah'] = body_df['capacity_dchg_Ah'] / 1000
        body_df['energy_chg_Wh'] = body_df['energy_chg_Wh'] / 1000
        body_df['energy_dchg_Wh'] = body_df['energy_dchg_Wh'] / 1000

    body_df['timestamp'] = pd.to_datetime(body_df[['year', 'month', 'day', 'hour', 'minute', 'second']], format='%Y-%m-%d %H:%M:%S')
    body_df = body_df.drop(['year', 'month', 'day', 'hour', 'minute', 'second'], axis=1)

    name_condtions = [
        body_df['step_name_raw'] == 1,
        body_df['step_name_raw'] == 2,
        body_df['step_name_raw'] == 3,
        body_df['step_name_raw'] == 4,
        body_df['step_name_raw'] == 5,
        body_df['step_name_raw'] == 6,
        body_df['step_name_raw'] == 7,
        body_df['step_name_raw'] == 8,
        body_df['step_name_raw'] == 9,
    ]
    name_ouputs = [
        'CC_Chg',
        'CC_Dchg',
        '3',
        'Rest',
        '5',
        '6',
        'CCCV_Chg',
        '8',
        '9',
    ]
    body_df['step_name'] = np.select(name_condtions, name_ouputs, '0')
    body_df = body_df.drop(['step_name_raw'], axis=1)

    main_df = body_df[body_df['aux_indicator'] == 85]
    auxt_df = body_df[body_df['aux_indicator'] == 357]

    # Recalculate record_id, step_id, and cycle_id, then drop columns
    main_df = main_df.drop_duplicates(subset=['record_raw'])
    main_df = main_df.sort_values(by=['record_raw']).reset_index(drop=True)
    main_df['record_id'] = np.arange(main_df.shape[0]) + 1
    main_df.set_index(['record_id'], inplace=True)

    main_df['step_id'] = (main_df['step_method'] != main_df['step_method'].shift(1)).cumsum()

    if not debug:
        main_df = main_df.drop(['aux_indicator', 'record_raw', 'step_raw', 'cycle_raw', 'temp_C', 'offset', 'current_range'], axis=1)

        main_cols = main_df.columns.tolist()
        main_cols = main_cols[-2:] + main_cols[-3:-2] + main_cols[:-3]
        main_df = main_df[main_cols]

    if len(auxt_df) > 0:
        auxt_df = auxt_df.drop_duplicates(subset=['record_raw'])
        auxt_df = auxt_df.sort_values(by=['record_raw']).reset_index(drop=True)
        auxt_df['record_id'] = np.arange(auxt_df.shape[0]) + 1
        auxt_df.set_index(['record_id'], inplace=True)

        auxt_df['step_id'] = (auxt_df['step_method'] != auxt_df['step_method'].shift(1)).cumsum()

        if not debug:
            auxt_df = auxt_df.drop(['aux_indicator', 'record_raw', 'step_raw', 'cycle_raw', 'capacity_chg_Ah',
                                    'capacity_dchg_Ah', 'energy_chg_Wh', 'energy_dchg_Wh', 'offset', 'current_range'], axis=1)

            auxt_cols = auxt_df.columns.tolist()
            auxt_cols = auxt_cols[-2:] + auxt_cols[-3:-2] + auxt_cols[:-3]
            auxt_df = auxt_df[auxt_cols]

    else:
        auxt_df = None

    return [main_df, auxt_df]


def read_file(inpath, debug=False):
    starttime = time.time()

    with open(inpath, "rb") as f:
        header_data = f.read()
        header_size = header_data.find(b'U\x00\x01')

    meta_data = process_header(header_data)

    with open(inpath, "rb") as f:
        f.seek(header_size, os.SEEK_SET)
        body_data = f.read()

    body_data_2 = process_body_bytes(body_data, debug)
    body_np = process_body_np(body_data_2, meta_data['current_limit'], debug)
    raw_data, auxt_data = process_body_df(body_np, meta_data['current_limit'], debug)

    meta_data['capacity_max_Ah'] = raw_data['capacity_dchg_Ah'].max()
    meta_data['voltage_upper_limit'] = raw_data['voltage_V'].max()
    meta_data['voltage_lower_limit'] = raw_data['voltage_V'].min()

    endtime = time.time()
    print(f'{os.path.basename(inpath)}...{round(endtime-starttime, 3)} s')

    if debug:
        return{'meta_data': meta_data, 'raw_data': raw_data, 'auxt_data': auxt_data, 'np_array': body_np}
    else:
        return {'meta_data': meta_data, 'raw_data': raw_data, 'auxt_data': auxt_data}


if __name__ == '__main__':
    inpath = r'C:\Users\Thomas Moran\Desktop\data_analysis\TMC19A1H001FM1.nda'
    # ec_data = read_file(inpath, debug=False)
    # data_np = process_body_np(ec_data, 6000)
    # date_df = process_body_df(data_np)
