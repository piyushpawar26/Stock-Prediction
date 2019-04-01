import MySQLdb

def connection():
	conn = MySQLdb.connect(host="localhost", user="root", passwd="root", db="sad")
	cur = conn.cursor()
	return cur, conn 