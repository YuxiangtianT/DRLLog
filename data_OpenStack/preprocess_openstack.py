import os
import sys
import logging
import pandas as pd
from spellpy import spell
# import myspell

logging.basicConfig(level=logging.WARNING, format='[%(asctime)s][%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


def deeplog_df_transfer(df, event_id_map):
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df[['datetime', 'EventId']]
    df['EventId'] = df['EventId'].apply(lambda e: event_id_map[e] if event_id_map.get(e) else -1)
    deeplog_df = df.set_index('datetime').resample('1min').apply(_custom_resampler).reset_index()
    return deeplog_df


def _custom_resampler(array_like):
    return list(array_like)


def deeplog_file_generator(filename, df):
    with open(filename, 'w') as f:
        for event_id_list in df['EventId']:
            for event_id in event_id_list:
                f.write(str(event_id) + ' ')
            f.write('\n')


if __name__ == '__main__':
    ##########
    # Parser #
    ##########
    input_dir = ''
    output_dir = 'openstack_result1/'
    log_format = '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>'
    log_main = 'open_stack'
    tau = 0.5

    parser = spell.LogParser(
        indir=input_dir,
        outdir=output_dir,
        log_format=log_format,
        logmain=log_main,
        tau=tau,
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for log_name in ['openstack_abnormal.log', 'openstack_normal2.log', 'openstack_normal1.log']:
        parser.parse(log_name)

    # parser.parse()

