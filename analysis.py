from pathlib import Path

from .read_nda import read_file
import os
import time
import numpy as np
import pandas as pd


class Dataset:

    def __init__(self, file_paths, active_mass_g=None, design_capacity_Ah=None, rated_capacity_Ah=None, upper_voltage_limit=None, lower_voltage_limit=None):
        file_paths = [file_paths] if isinstance(file_paths, str) else file_paths
        self.file_paths = file_paths

        ec_data_dict = {}
        for file_path in file_paths:
            ec_data_dict[os.path.basename(file_path)] = read_file(file_path)

        self.ec_data = ec_data_dict
        self.ec_data_display = ec_data_dict

        last_file_name = list(self.ec_data.items())[-1]

        self.active_mass_g = active_mass_g if active_mass_g else self.ec_data[last_file_name[0]]['meta_data']['active_mass_g']
        self.design_capacity_Ah = design_capacity_Ah if design_capacity_Ah else round(self.ec_data[last_file_name[0]]['meta_data']['capacity_max_Ah'], 1)
        self.rated_capacity_Ah = rated_capacity_Ah if rated_capacity_Ah else self.ec_data[last_file_name[0]]['meta_data']['capacity_max_Ah']
        self.voltage_upper_limit = upper_voltage_limit if upper_voltage_limit else self.ec_data[last_file_name[0]]['meta_data']['voltage_upper_limit']
        self.voltage_lower_limit = lower_voltage_limit if lower_voltage_limit else self.ec_data[last_file_name[0]]['meta_data']['voltage_lower_limit']

        # self.analyze()

    def merge(self):
        last_key = list(self.ec_data.keys())[-1]
        meta_data = self.ec_data[last_key]['meta_data']
        raw_df_list = [None] * len(self.ec_data)
        auxt_df_list = [None] * len(self.ec_data)

        file_data_list = []
        for file_name, file_data in self.ec_data.items():
            file_data_list.append({
                'file_name': file_name,
                'file_data': file_data,
                'date_start': file_data['raw_data']['timestamp'].iloc[0],
            })
        file_data_list.sort(key=lambda x: x['date_start'])

        for i, list_item in enumerate(file_data_list):
            raw_df_list[i] = list_item['file_data']['raw_data']

            if 'auxt_data' in list_item['file_data'].keys():
                auxt_df_list[i] = list_item['file_data']['auxt_data']


        raw_data = pd.concat(raw_df_list, ignore_index=True)
        raw_data = raw_data.sort_values(by=['timestamp'], )

        raw_data = raw_data.reset_index()
        raw_data['record_id'] = raw_data['index'] + 1
        del raw_data['index']
        raw_data = raw_data.sort_values(by='record_id')
        raw_data['step_id'] = (raw_data['step_method'] != raw_data['step_method'].shift(1)).cumsum()

        if not any(elem is None for elem in auxt_df_list):
            auxt_data = pd.concat(auxt_df_list, ignore_index=True)
            auxt_data = auxt_data.sort_values(by=['timestamp'])
            auxt_data = auxt_data.reset_index()
            auxt_data['record_id'] = auxt_data['index'] + 1
            del auxt_data['index']
            auxt_data = auxt_data.sort_values(by='record_id')
            auxt_data['step_id'] = (auxt_data['step_method'] != auxt_data['step_method'].shift(1)).cumsum()
        else:
            auxt_data = None

        self.ec_data['merged_data'] = {'meta_data': meta_data, 'raw_data': raw_data, 'auxt_data': auxt_data}
        self.ec_data_display['merged_data'] = {'meta_data': meta_data, 'raw_data': raw_data, 'auxt_data': auxt_data}

        return

    def filter_step_data(self, filters):
        for file_name, file_data in self.ec_data.items():
            step_data = file_data['step_data']

            if 'step_id' in filters.keys():
                lower_limit = filters['step_id'][0]
                upper_limit = filters['step_id'][1]
                step_data = step_data[step_data['step_id'].between(lower_limit, upper_limit)]

            if 'step_time' in filters.keys():
                lower_limit = filters['step_time'][0]
                upper_limit = filters['step_time'][1]
                step_data = step_data[step_data['step_time_m'].between(lower_limit, upper_limit)]

            if 'current_i' in filters.keys():
                lower_limit = filters['current_i'][0]
                upper_limit = filters['current_i'][1]
                step_data = step_data[abs(step_data['current_i_A']).between(lower_limit, upper_limit)]

            if 'current_f' in filters.keys():
                lower_limit = filters['current_f'][0]
                upper_limit = filters['current_f'][1]
                step_data = step_data[(step_data['current_f_A'].between(lower_limit, upper_limit)) | (step_data['current_f_A'].between(-upper_limit, -lower_limit))]

            if 'current_avg' in filters.keys():
                lower_limit = filters['current_avg'][0]
                upper_limit = filters['current_avg'][1]
                step_data = step_data[(step_data['current_avg_A'].between(lower_limit, upper_limit)) | (step_data['current_avg_A'].between(-upper_limit, -lower_limit))]

            if 'voltage_i' in filters.keys():
                lower_limit = filters['voltage_i'][0]
                upper_limit = filters['voltage_i'][1]
                step_data = step_data[step_data['voltage_i_V'].between(lower_limit, upper_limit)]

            if 'voltage_f' in filters.keys():
                lower_limit = filters['voltage_f'][0]
                upper_limit = filters['voltage_f'][1]
                step_data = step_data[step_data['voltage_f_V'].between(lower_limit, upper_limit)]

            if 'voltage_avg' in filters.keys():
                lower_limit = filters['voltage_avg'][0]
                upper_limit = filters['voltage_avg'][1]
                step_data = step_data[step_data['voltage_avg_V'].between(lower_limit, upper_limit)]

            if 'capacity' in filters.keys():
                lower_limit = filters['capacity'][0]
                upper_limit = filters['capacity'][1]
                step_data = step_data[(step_data['capacity_chg_Ah'].between(lower_limit, upper_limit)) | (step_data['capacity_dchg_Ah'].between(lower_limit, upper_limit))]

            self.ec_data_display[file_name]['step_data'] = step_data

        return

    def filter_cycle_data(self, filters):
        for file_name, file_data in self.ec_data.items():
            cycle_data = file_data['cycle_data']

            if 'cycle_id' in filters.keys():
                lower_limit = filters['cycle_id'][0]
                upper_limit = filters['cycle_id'][1]
                cycle_data = cycle_data[cycle_data['cycle_id'].between(lower_limit, upper_limit)]

            if 'cycle_time' in filters.keys():
                lower_limit = filters['cycle_time'][0]
                upper_limit = filters['cycle_time'][1]
                cycle_data = cycle_data[cycle_data['step_time_h'].between(lower_limit, upper_limit)]

            if 'capacity_chg' in filters.keys():
                lower_limit = filters['capacity_chg'][0]
                upper_limit = filters['capacity_chg'][1]
                cycle_data = cycle_data[cycle_data['capacity_chg_Ah'].between(lower_limit, upper_limit)]

            if 'capacity_dchg' in filters.keys():
                lower_limit = filters['capacity_dchg'][0]
                upper_limit = filters['capacity_dchg'][1]
                cycle_data = cycle_data[cycle_data['capacity_dchg_Ah'].between(lower_limit, upper_limit)]

            if 'columbic_eff' in filters.keys():
                lower_limit = filters['columbic_eff'][0]
                upper_limit = filters['columbic_eff'][1]
                cycle_data = cycle_data[cycle_data['columbic_eff'].between(lower_limit, upper_limit)]

            if 'normalized_dchg' in filters.keys():
                lower_limit = filters['normalized_dchg'][0]
                upper_limit = filters['normalized_dchg'][1]
                cycle_data = cycle_data[cycle_data['normalized_dchg'].between(lower_limit, upper_limit)]

            self.ec_data_display[file_name]['step_data'] = cycle_data

        return

    def change_units(self, convert_to_mA=True):
        for file_name, file_data in self.ec_data.items():
            raw_data = file_data['raw_data']

            if convert_to_mA and ('current_A' in raw_data.columns.tolist()):
                raw_data['current_mA'] = raw_data['current_A'] * 1000
                raw_data['capacity_chg_mAh'] = raw_data['capacity_chg_Ah'] * 1000
                raw_data['capacity_dchg_mAh'] = raw_data['capacity_dchg_Ah'] * 1000
                raw_data['energy_chg_mWh'] = raw_data['energy_chg_Wh'] * 1000
                raw_data['energy_dchg_mWh'] = raw_data['energy_dchg_Wh'] * 1000
                raw_data = raw_data.drop(['current_A', 'capacity_chg_Ah', 'capacity_dchg_Ah', 'energy_chg_Wh', 'energy_dchg_Wh'], axis=1)
                self.ec_data_display[file_name]['raw_data'] = raw_data

                if 'step_data' in file_data.keys():
                    step_data = file_data['step_data']
                    step_data['capacity_chg_mAh'] = step_data['capacity_chg_Ah'] * 1000
                    step_data['capacity_dchg_mAh'] = step_data['capacity_dchg_Ah'] * 1000
                    step_data['energy_chg_mWh'] = step_data['energy_chg_Wh'] * 1000
                    step_data['energy_dchg_mWh'] = step_data['energy_dchg_Wh'] * 1000
                    step_data['current_i_mA'] = step_data['current_i_A'] * 1000
                    step_data['current_f_mA'] = step_data['current_f_A'] * 1000
                    step_data['current_avg_mA'] = step_data['current_avg_A'] * 1000
                    step_data['resistance_mohm'] = step_data['resistance_ohm'] * 1000
                    step_data = step_data.drop(['capacity_chg_Ah', 'capacity_dchg_Ah', 'energy_chg_Wh', 'energy_dchg_Wh', 'current_i_A', 'current_f_A', 'current_avg_A', 'resistance_ohm'], axis=1)
                    self.ec_data_display[file_name]['step_data'] = step_data

                if 'cycle_data' in file_data.keys():
                    cycle_data = file_data['cycle_data']
                    cycle_data['capacity_chg_mAh'] = cycle_data['capacity_chg_Ah'] * 1000
                    cycle_data['capacity_dchg_mAh'] = cycle_data['capacity_dchg_Ah'] * 1000
                    cycle_data['energy_chg_mWh'] = cycle_data['energy_chg_Wh'] * 1000
                    cycle_data['energy_dchg_mWh'] = cycle_data['energy_dchg_Wh'] * 1000
                    cycle_data = cycle_data.drop(['capacity_chg_Ah', 'capacity_dchg_Ah', 'energy_chg_Wh', 'energy_dchg_Wh'], axis=1)
                    self.ec_data_display[file_name]['cycle_data'] = cycle_data

            if not convert_to_mA and ('current_mA' in raw_data.columns.tolist()):
                raw_data['current_A'] = raw_data['current_mA'] / 1000
                raw_data['capacity_chg_Ah'] = raw_data['capacity_chg_mAh'] / 1000
                raw_data['capacity_dchg_Ah'] = raw_data['capacity_dchg_mAh'] / 1000
                raw_data['energy_chg_Wh'] = raw_data['energy_chg_mWh'] / 1000
                raw_data['energy_dchg_Wh'] = raw_data['energy_dchg_mWh'] / 1000
                raw_data = raw_data.drop(['current_mA', 'capacity_chg_mAh', 'capacity_dchg_mAh', 'energy_chg_mWh', 'energy_dchg_mWh'], axis=1)
                self.ec_data_display[file_name]['raw_data'] = raw_data

                if 'step_data' in file_data.keys():
                    step_data = file_data['step_data']
                    step_data['capacity_chg_Ah'] = step_data['capacity_chg_mAh'] / 1000
                    step_data['capacity_dchg_Ah'] = step_data['capacity_dchg_mAh'] / 1000
                    step_data['energy_chg_Wh'] = step_data['energy_chg_mWh'] / 1000
                    step_data['energy_dchg_Wh'] = step_data['energy_dchg_mWh'] / 1000
                    step_data['current_i_A'] = step_data['current_i_mA'] / 1000
                    step_data['current_f_A'] = step_data['current_f_mA'] / 1000
                    step_data['current_avg_A'] = step_data['current_avg_mA'] / 1000
                    step_data['resistance_ohm'] = step_data['resistance_mohm'] / 1000
                    step_data = step_data.drop(['capacity_chg_mAh', 'capacity_dchg_mAh', 'energy_chg_mWh', 'energy_dchg_mWh', 'current_i_mA', 'current_f_mA', 'current_avg_mA', 'resistance_mohm'], axis=1)
                    self.ec_data_display[file_name]['step_data'] = step_data

                if 'cycle_data' in file_data.keys():
                    cycle_data = file_data['cycle_data']
                    cycle_data['capacity_chg_Ah'] = cycle_data['capacity_chg_mAh'] / 1000
                    cycle_data['capacity_dchg_Ah'] = cycle_data['capacity_dchg_mAh'] / 1000
                    cycle_data['energy_chg_Wh'] = cycle_data['energy_chg_mWh'] / 1000
                    cycle_data['energy_dchg_Wh'] = cycle_data['energy_dchg_mWh'] / 1000
                    cycle_data = cycle_data.drop(['capacity_chg_mAh', 'capacity_dchg_mAh', 'energy_chg_mWh', 'energy_dchg_mWh'], axis=1)
                    self.ec_data_display[file_name]['cycle_data'] = cycle_data

            return self.ec_data_display

    def raw_data(self):
        results = {}
        for file_name, file_data in self.ec_data_display.items():
            results[file_name] = file_data['raw_data']

        return results

    def all_data(self):
        return self.ec_data_dispaly

    def calc_step_data(self, add_cycle=True):
        for file_name, file_data in self.ec_data.items():
            raw_data = file_data['raw_data']
            step_data = pd.DataFrame({
                'step_id': raw_data.groupby('step_id')['step_id'].first(),
                'step_name': raw_data.groupby('step_id')['step_name'].first(),
                'step_method': raw_data.groupby('step_id')['step_method'].first(),
                'step_time_m': raw_data.groupby('step_id')['step_time_s'].max() / 60,
                'capacity_chg_Ah': raw_data.groupby('step_id')['capacity_chg_Ah'].max(),
                'capacity_dchg_Ah': raw_data.groupby('step_id')['capacity_dchg_Ah'].max(),
                'energy_chg_Wh': raw_data.groupby('step_id')['energy_chg_Wh'].max(),
                'energy_dchg_Wh': raw_data.groupby('step_id')['energy_dchg_Wh'].max(),
                'voltage_i_V': raw_data.groupby('step_id')['voltage_V'].first(),
                'voltage_f_V': raw_data.groupby('step_id')['voltage_V'].last(),
                'voltage_avg_V': raw_data.groupby('step_id')['voltage_V'].mean(),
                'current_i_A': raw_data.groupby('step_id')['current_A'].first(),
                'current_f_A': raw_data.groupby('step_id')['current_A'].last(),
                'current_avg_A': raw_data.groupby('step_id')['current_A'].mean(),
                'timestamp_i': raw_data.groupby('step_id')['timestamp'].first(),
                'timestamp_f': raw_data.groupby('step_id')['timestamp'].last(),
            })
            step_data['voltage_drop_V'] = step_data['voltage_f_V'] - step_data['voltage_f_V'].shift(1)
            step_data['resistance_ohm'] = step_data['voltage_drop_V'] / step_data['current_i_A']
            step_data.set_index(['step_id'], inplace=True)

            # Calculate cycle_id from step_data
            cycle_list = [None] * len(step_data.index)
            cycle = 1
            check = 0
            i = 0
            for id, step in step_data.iterrows():
                if step['step_name'] == 'CC_Dchg':
                    check = 1
                if (step['step_name'] == 'CC_Chg' or step['step_name'] == 'CCCV_Chg') and check == 1:
                    cycle += 1
                    check = 0
                cycle_list[i] = cycle
                i += 1
            step_data['cycle_id'] = cycle_list

            if add_cycle:
                # Add cycle_ids to raw_data and auxt_data - SLOW!
                step_list = step_data.index.tolist()
                cycle_conditions = [None] * len(step_data)
                j = 0
                for step in step_list:
                    cycle_conditions[j] = raw_data['step_id'] == step
                    j += 1

                raw_data['cycle_id'] = np.select(cycle_conditions, cycle_list, 1)
                raw_cols = raw_data.columns.tolist()
                # raw_cols = raw_cols[:2] + raw_cols[-1:] + raw_cols[2:-1]
                self.ec_data[file_name]['raw_data'] = raw_data[raw_cols]

                auxt_data = self.ec_data[file_name]['auxt_data']
                if auxt_data is not None:
                    auxt_data['cycle_id'] = np.select(cycle_conditions, cycle_list, 1)
                    auxt_cols = auxt_data.columns.tolist()
                    # auxt_cols = auxt_cols[:2] + auxt_cols[-1:] + auxt_cols[2:-1]
                    self.ec_data[file_name]['auxt_data'] = auxt_data[auxt_cols]

            step_cols = step_data.columns.tolist()
            step_cols = step_cols[:1] + step_cols[-1:] + step_cols[1:-1]
            step_data = step_data[step_cols]

            self.ec_data[file_name]['step_data'] = step_data
            self.ec_data_display[file_name]['step_data'] = step_data

        return

    def calc_cycle_data(self):
        for file_name, file_data in self.ec_data.items():
            if 'step_data' not in file_data.keys():
                continue
            step_data = file_data['step_data']
            cycle_data = pd.DataFrame({
                'cycle_id': step_data.groupby('cycle_id')['cycle_id'].first(),
                'cycle_time_h': step_data.groupby('cycle_id')['step_time_m'].sum() / 60,
                'capacity_chg_Ah': step_data.groupby('cycle_id')['capacity_chg_Ah'].sum(),
                'capacity_dchg_Ah': step_data.groupby('cycle_id')['capacity_dchg_Ah'].sum(),
                'energy_chg_Wh': step_data.groupby('cycle_id')['energy_chg_Wh'].sum(),
                'energy_dchg_Wh': step_data.groupby('cycle_id')['energy_dchg_Wh'].sum(),
            })
            cycle_data['columbic_eff'] = cycle_data['capacity_dchg_Ah'] / cycle_data['capacity_chg_Ah']
            cycle_data['normalized_dchg'] = cycle_data['capacity_dchg_Ah'] / self.rated_capacity_Ah
            mass_g = self.active_mass_g if self.active_mass_g != 0 else 1
            cycle_data['specific_chg_mAhg'] = cycle_data['capacity_chg_Ah'] * 1000 / mass_g
            cycle_data['specific_dchg_mAhg'] = cycle_data['capacity_dchg_Ah'] * 1000 / mass_g
            cycle_data.set_index(['cycle_id'],inplace=True)

            self.ec_data[file_name]['cycle_data'] = cycle_data
            self.ec_data_display[file_name]['cycle_data'] = cycle_data

        return

    def analyze(self):
        self.calc_step_data()
        self.calc_cycle_data()
        return self.ec_data

    def export_csv(self, output_path):
        for file_name, file_data in self.ec_data.items():
            file_name_clean = file_name[:-4] if len(file_name) >= 14 else file_name
            for table_name, table_data in file_data.items():
                file_path = r'{}\{}'.format(output_path, file_name_clean)
                Path(file_path).mkdir(parents=True, exist_ok=True)
                if table_name == 'meta_data':
                    with open(r'{}\{}.csv'.format(file_path, table_name), 'w') as f:
                        for data in table_data.keys():
                            f.write("%s,%s\n"%(data,table_data[data]))
                else:
                    if table_data is None:
                        continue

                    if not os.path.exists(file_path):
                        os.makedirs(file_path)

                    table_data.to_csv(r'{}\{}.csv'.format(file_path, table_name), index=True)

        return

    def export_excel(self, output_path):
        self.change_units(convert_to_mA=True)
        for file_name, file_data in self.ec_data_display.items():
            file_name_clean = file_name[:-4] if len(file_name) >= 14 else file_name
            channel = f"{file_data['meta_data']['machine_id']}_{file_data['meta_data']['row_id']}_{file_data['meta_data']['channel_id']}"
            raw_data = file_data['raw_data']
            detail = pd.DataFrame({
                'Record number': raw_data.index,
                'status': raw_data['step_name'],
                'Jump': raw_data['step_id'] + 1,
                'Cycle': raw_data['cycle_id'],
                'Steps': raw_data['step_id'],
                'Current(mA)': raw_data['current_mA'],
                'Voltage(V)': raw_data['voltage_V'],
                'Capacity(mAh)': raw_data['capacity_chg_mAh'] + raw_data['capacity_dchg_mAh'],
                'Energy(mWh)': raw_data['energy_chg_mWh'] + raw_data['energy_dchg_mWh'],
                'Relative Time(h:min:s.ms)': pd.to_datetime(raw_data['step_time_s'], unit='s'),
                'Real Time(h:min:s.ms)': raw_data['timestamp'],
            })
            detail.set_index(['Record number'], inplace=True)
            detail['Relative Time(h:min:s.ms)'] = detail['Relative Time(h:min:s.ms)'].dt.strftime('%H:%M:%S.%f')

            with pd.ExcelWriter(r'{}\{}.xlsx'.format(output_path, file_name_clean)) as workbook:
                for table_name, table_data in file_data.items():
                    if table_name == 'meta_data':
                        continue

                    if table_name == 'raw_data':
                        detail.to_excel(workbook, sheet_name=f'Detail_{channel}')

                    if table_name == 'step_data':
                        table_data.to_excel(workbook, sheet_name=f'Statis_{channel}')

                    if table_name == 'cycle_data':
                        table_data.to_excel(workbook, sheet_name=f'Cycle_{channel}')
        return


def bulk_load(data_dir, battery_df, merge=False, steps_filters=None, cycle_filters=None, change_units=False):
    grouped_file_paths = {}
    all_data = {}
    for root, subdirs, files in os.walk(data_dir):
        for filename in files:
            battery_id = filename[:11]
            file_path = os.path.join(root, filename)
            if filename[-3:] == 'nda' and battery_id in battery_df.index:
                if battery_id in grouped_file_paths:
                    grouped_file_paths[battery_id]['file_paths'].append(file_path)
                else:
                    grouped_file_paths[battery_id] = {
                        'active_mass': battery_df.loc[battery_id, 'Active Mass (g)'],
                        'group_name': battery_df.loc[battery_id, 'Group Name'],
                        'test_plan': battery_df.loc[battery_id, 'Test Plan'],
                        'file_paths': [file_path, ]
                    }

    for battery_id, battery_data in grouped_file_paths.items():
        my_battery = Dataset(battery_data['file_paths'],  active_mass_g=battery_data['active_mass'])

        if merge:
            my_battery.merge()

        my_battery.analyze()

        if steps_filters:
            my_battery.filter_step_data(steps_filters)

        if cycle_filters:
            my_battery.filter_cycle_data(cycle_filters)

        if change_units:
            my_battery.change_units()

        all_data[battery_id] = my_battery.ec_data_display

    return all_data

if __name__ == '__main__':
    start = time.time()
    my_file_path = r'D:\Lionano\Lionano Team Site - Battery Engineering\Cell Testing\Material Evaluation\Raw Data\Cath Eval-19 (Process Improvement)\Cycling - 25C - 0.33C\TMC19A1H004RC4.nda'
    data = Dataset(my_file_path)
    data.analyze()
    data.export_csv(r'C:\Users\Thomas Moran\Desktop\data_output')
    end = time.time()
    print(round(end-start, 3))
