# import os
# import time
# import socket
# import inspect
# import logging
# import datetime
# import schedule
# import win32event
# import win32service
# import win32serviceutil
from data_processing import write_stocks_probability_to_db


# region win32serviceutil.HandleCommandLine(SimulatedTransaction)
# class SimulatedTransaction(win32serviceutil.ServiceFramework):
#     # 服务名
#     _svc_name_ = "SimulatedTransaction"
#     # 服务在windows系统中显示的名称
#     _svc_display_name_ = "SimulatedTransaction"
#     # 服务的描述
#     _svc_description_ = "为聚宽平台模拟交易提供的数据服务"

#     def __init__(self, args):
#         win32serviceutil.ServiceFramework.__init__(self, args)
#         self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
#         socket.setdefaulttimeout(60)
#         self.is_run = True
#         self.logger = self._get_logger()
    
#     @staticmethod
#     def _get_logger():
#         logger = logging.getLogger('[SimulatedTransaction]')
#         this_file = inspect.getfile(inspect.currentframe())
#         dir_path = os.path.abspath(os.path.dirname(this_file))
#         handler = logging.FileHandler(os.path.join(dir_path, "service.log"))
#         formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
#         handler.setFormatter(formatter)
#         logger.addHandler(handler)
#         logger.setLevel(logging.INFO)
#         return logger

#     def job(self):
#         self.logger.info('executing job....')
#         msg = write_stocks_probability_to_db()
#         self.logger.info(msg)
    
#     def start_run(self):
#         while self.is_run:
#             schedule.run_pending()
#             time.sleep(1)

#     def SvcDoRun(self):
#         self.logger.info("service is run....")
#         # schedule.every().seconds.do(self.job)
#         schedule.every().day.at("16:30").do(self.job)
#         self.start_run()
#         self.ReportServiceStatus(win32service.SERVICE_RUNNING)
#         win32event.WaitForSingleObject(self.hWaitStop, win32event.INFINITE)

#     def SvcStop(self):
#         self.logger.info("service is stop....")
#         self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
#         win32event.SetEvent(self.hWaitStop)
#         self.is_run = False
# endregion


# region plan 2
# def _get_logger():
#     logger = logging.getLogger('[PythonService]')

#     this_file = inspect.getfile(inspect.currentframe())
#     dir_path = os.path.abspath(os.path.dirname(this_file))
#     handler = logging.FileHandler(os.path.join(dir_path, "service.log"))

#     formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
#     handler.setFormatter(formatter)

#     logger.addHandler(handler)
#     logger.setLevel(logging.INFO)
#     return logger


# logger = _get_logger()


# def job():
#     logger.info("job start....")
#     msg = write_stocks_probability_to_db()
#     logger.info(msg)
#     logger.info("job end....")


# def timed_task():
#     logger.info("service is run....")
#     # schedule.every().seconds.do(job)
#     schedule.every().day.at("08:05").do(job)
#     while True:
#         print("【"+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+"】持续监测中....")
#         schedule.run_pending()
#         time.sleep(1)
# endregion


if __name__ == "__main__":
    # timed_task()·
    print(write_stocks_probability_to_db())
    print("====complete====")