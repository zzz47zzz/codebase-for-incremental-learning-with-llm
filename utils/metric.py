from typing import Any
import numpy as np 

class ResultSummary(object):

    # ============================================================================
    # Result Summary Matrix
    #                           Task 0   Task 1   Task 2
    #                        -----------------------------
    #  Learning After Task 0 |        |         |        |
    #                        -----------------------------
    #  Learning After Task 1 |        |         |        |
    #                        -----------------------------
    #  Learning After Task 2 |        |         |        |
    #                        -----------------------------
    # ============================================================================

    def __init__(self, num_task: int) -> None:
        self.result_summary = np.ones((num_task,num_task))*-1

    def update(self, task_id: int, eval_task_id: int, value: float) -> None:
        '''
            Update the metric:

            self.result_summary[task_id,eval_task_id] = value

            Args:
                - task_id: evaluation after learning the {task_id}-th task
                - eval_task_id: the evaluation result of the {task_id}-th task
                - value: the value to be updated
        '''
        self.result_summary[task_id,eval_task_id] = value

    def get_value(self) -> np.array:

        return self.result_summary

    def print_format(self, round: int=2) -> np.array:

        return np.around(self.result_summary,round)