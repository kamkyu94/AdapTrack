import numpy as np
import trackeval.metrics as metrics
from trackeval.eval import Evaluator
import trackeval.datasets as datasets


def evaluate(setting):
    # Set evaluation configurations
    eval_config = {'USE_PARALLEL': True,
                   'NUM_PARALLEL_CORES': 8,
                   'BREAK_ON_ERROR': True,
                   'RETURN_ON_ERROR': False,
                   'LOG_ON_ERROR': '../outputs/error_log.txt',

                   'PRINT_RESULTS': False,
                   'PRINT_ONLY_COMBINED': False,
                   'PRINT_CONFIG': False,
                   'TIME_PROGRESS': False,
                   'DISPLAY_LESS_PROGRESS': True,

                   'OUTPUT_SUMMARY': False,
                   'OUTPUT_EMPTY_CLASSES': False,
                   'OUTPUT_DETAILED': False,
                   'PLOT_CURVES': False}

    dataset_config = {'GT_FOLDER': setting['gt_folder'],
                      'TRACKERS_FOLDER': setting['trackers_folder'],
                      'OUTPUT_FOLDER': None,
                      'TRACKERS_TO_EVAL': [setting['tracker']],
                      'CLASSES_TO_EVAL': ['pedestrian'],
                      'BENCHMARK': setting['dataset'],
                      'SPLIT_TO_EVAL': 'val',
                      'INPUT_AS_ZIP': False,
                      'PRINT_CONFIG': False,
                      'DO_PREPROC': True,
                      'TRACKER_SUB_FOLDER': '',
                      'OUTPUT_SUB_FOLDER': '',
                      'TRACKER_DISPLAY_NAMES': None,
                      'SEQMAP_FOLDER': None,
                      'SEQMAP_FILE': './trackeval/seqmap/%s/val.txt' % setting['dataset'].lower(),
                      'SEQ_INFO': None,
                      'GT_LOC_FORMAT': setting['gt_loc_format'],
                      'SKIP_SPLIT_FOL': True}

    # Set configuration
    evaluator = Evaluator(eval_config)
    dataset_list = [datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [metrics.HOTA(), metrics.CLEAR(), metrics.Identity()]
    res, _ = evaluator.evaluate(dataset_list, metrics_list)

    # Get
    hota = np.mean(res['MotChallenge2DBox'][setting['tracker']]['COMBINED_SEQ']['pedestrian']['HOTA']['HOTA']).item()
    deta = np.mean(res['MotChallenge2DBox'][setting['tracker']]['COMBINED_SEQ']['pedestrian']['HOTA']['DetA']).item()
    assa = np.mean(res['MotChallenge2DBox'][setting['tracker']]['COMBINED_SEQ']['pedestrian']['HOTA']['AssA']).item()
    mota = res['MotChallenge2DBox'][setting['tracker']]['COMBINED_SEQ']['pedestrian']['CLEAR']['MOTA']
    idf1 = res['MotChallenge2DBox'][setting['tracker']]['COMBINED_SEQ']['pedestrian']['Identity']['IDF1']

    # Print
    print('%.3f %.3f %.3f %.3f %.3f' % (hota * 100, idf1 * 100, assa * 100, mota * 100, deta * 100), flush=True)
