import pymssql
import pymysql
from pymysql.cursors import DictCursor


class SqlServer:
    def __init__(self):
        self.server = "127.0.0.1"
        self.user = "sa"
        self.password = "123"
        self.database = "AI.Projects"
        self.charset = "utf8"

    def conn_server(self):
        conn = pymssql.connect(self.server, self.user, self.password, self.database, charset=self.charset)
        cursor = conn.cursor(as_dict=True)
        return conn, cursor

    def get_sql(self, sql):
        conn, cursor = self.conn_server()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        conn.close()
        return result

    def execute_sql(self, sql):
        conn, cursor = self.conn_server()
        cursor.execute(sql)
        conn.commit()
        cursor.close()
        conn.close()
        

class MySql:
    def __init__(self):
        self.server = "rm-wz9hmzrjuip6f1mtbvo.mysql.rds.aliyuncs.com"
        self.user = "zhaoxinchen"
        self.password = "Abcd5678!"
        self.database = "xinchen"
        self.charset = "utf8"

    def conn_server(self):
        conn = pymysql.connect(
            host=self.server, 
            user=self.user, 
            password=self.password, 
            database=self.database, 
            charset=self.charset,
            cursorclass=DictCursor)
        cursor = conn.cursor()
        return conn, cursor

    def get_sql(self, sql):
        conn, cursor = self.conn_server()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        conn.close()
        return result

    def execute_sql(self, sql):
        conn, cursor = self.conn_server()
        cursor.execute(sql)
        conn.commit()
        cursor.close()
        conn.close()
