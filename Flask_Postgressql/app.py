import os
import psycopg2
from flask import Flask, render_template, request, redirect, url_for, flash
import psycopg2 #pip install psycopg2 
import psycopg2.extras

conn = psycopg2.connect( #psycopg2 database adaptor for implementing python
        host="localhost",
        database="students",
        user='postgres',
        password='p@ssw0rd')
app = Flask(__name__)

#To display created table
@app.route('/')
def Index():
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)  #cursor is used to read the rows in the table 
    #A cursor that keeps a list of column name is DictCursor

    s = "SELECT * FROM students"
    cur.execute(s) # Execute the SQL
    list_users = cur.fetchall()#fetches a
    print(list_users)
    return render_template('index.html', list_users = list_users)
    
#add an extra row to the table
@app.route('/add_student', methods=['POST'])
def add_student():
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    if request.method == 'POST':
        fname = request.form['fname']
        lname = request.form['lname']
        email = request.form['email']
        print(email)
        cur.execute("INSERT INTO students (fname, lname, email) VALUES (%s,%s,%s)", (fname, lname, email))
        conn.commit()
        flash('Student Added successfully')
        return redirect(url_for('Index'))
 
@app.route('/edit/<id>', methods = ['POST', 'GET'])
def get_employee(id):
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
   
    cur.execute('SELECT * FROM students WHERE id = %s', (id))
    data = cur.fetchall()
    cur.close()
    print(data[0])
    return render_template('edit.html', student = data[0])

#update an row in the table
@app.route('/update/<id>', methods=['POST'])
def update_student(id):
    if request.method == 'POST':
        fname = request.form['fname']
        lname = request.form['lname']
        email = request.form['email']
         
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""
            UPDATE students
            SET fname = %s,
                lname = %s,
                email = %s
            WHERE id = %s
        """, (fname, lname, email, id))
        flash('Student Updated Successfully')
        conn.commit()
        return redirect(url_for('Index'))        
 
app.secret_key = 'the random string' 
if __name__ == "__main__":
    app.run(debug=True,port=5200)
   
