from flask import Flask, render_template, flash, request, redirect, url_for, session
from dbconnect import connection
from passlib.hash import sha256_crypt
from MySQLdb import escape_string as esc
import gc
from news_content import main_news, normal_news

app = Flask(__name__)
app.secret_key = 'my_unobvious_secret_key'

@app.route('/login/', methods=['GET', 'POST'])
def login():
	error = ''
	try:
		cur, conn = connection()
		if request.method == 'POST':
			if request.form['submit'] == 'login':
				attempted_username = str(request.form['eid'])
				attempted_password = str(request.form['pass'])
				query = "SELECT * FROM users WHERE email = '" + attempted_username + "';"
				cur.execute(query)
				password = cur.fetchone()[4]
				if sha256_crypt.verify(attempted_password, password):
					session['logged_in'] = True
					session['username'] = attempted_username
					return redirect(url_for('profile'))
				else:
					error = 'Invalid credentials.'
			elif request.form['submit'] == 'signup':
				fname = str(request.form['fname'])
				sname = str(request.form['sname'])
				email = str(request.form['email'])
				phone = str(request.form['phone'])
				password = str(request.form['pass'])
				repass = str(request.form['repass'])
				if fname and sname and email and phone and password and repass and len(phone) == 10 and password == repass:
					password = sha256_crypt.encrypt(password)
					cur.execute("INSERT INTO users (fname, surname, email, phone, password) VALUES (%s, %s, %s, %s, %s)", (esc(fname), esc(sname), esc(email), esc(phone), esc(password)))
					conn.commit()
					session['logged_in'] = True
					session['username'] = attempted_username
					return redirect(url_for('profile'))
				else:
					error = 'Invalid.'
		conn.close()
		cur.close()
		gc.collect()
		return render_template("login.html", error=error)
	except Exception as e:
		return str(e)

@app.route('/')
def index():
	main_news_list = main_news()
	normal_news_list = normal_news()
	return render_template("home.html", main_news=main_news_list, normal_news=normal_news_list)

@app.route('/profile/')
def profile():
	return render_template("profile.html")

if __name__=='__main__':
	app.run(debug=True)
