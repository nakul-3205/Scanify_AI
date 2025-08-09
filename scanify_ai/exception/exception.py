import sys
from scanify_ai.logging.log_config import logger

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number[{1}] error message [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error)
    )

class CustomException(Exception):
    def __init__(self,error_message,error_detail: sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)
    def __str__(self):
        return str(self.error_message)
    
# if __name__=='__main__':
#     try:
#         logger.info('entered try block')
#         a=1/0
#         print('not possible')
#     except Exception as e:
#         raise CustomException(e,sys)
