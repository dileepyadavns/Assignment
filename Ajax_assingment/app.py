import json
from flask import Flask, render_template, request, redirect, url_for, flash
import psycopg2 #pip install psycopg2 
import psycopg2.extras

conn = psycopg2.connect( #psycopg2 database adaptor for implementing python
        host="localhost",
        database="students",
        user='postgres',
        password='p@ssw0rd')
app = Flask(__name__)


@app.route('/jsono') #this link will show the json format of databse table
def rowdic():
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
   
    s = "SELECT * FROM student_details"
    cur.execute(s)
    res = cur.fetchall()
    list_rows = json.dumps([dict(r) for r in res])
    #json.dumps used to convert python object to json fromat
 
    return list_rows


#To Read created table
@app.route('/')
def Index():
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)  #cursor is used to read the rows in the table 
    #A cursor that keeps a list of column name is DictCursor

    s = "SELECT * FROM student_details"
    cur.execute(s) # Execute the SQL
    list_users = cur.fetchall()#fetches all the rows
    conn.commit()
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
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor) 
            cur.execute("""INSERT INTO student_details (fname, lname, email) VALUES
                (%s, %s,%s)""",(fname,lname,email))        
            conn.commit()
            

        except psycopg2.IntegrityError:
            conn.rollback()
            s = "SELECT * FROM student_details"
            cur.execute(s) # Execute the SQL
            list_users = cur.fetchall()#fetches all rows
            conn.commit()
            print(list_users)
            return render_template('index.html', list_users = list_users,msg='A user exists with same email try with new email')
    return redirect(url_for('Index'))
    
       
#to edit the conetents of table
@app.route('/edit/<id>', methods = ['POST', 'GET'])
def get_employee(id):
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor) 
    # DictCursor is a Cursor class that returns rows as dictionaries and stores the result set in the client
   
    cur.execute('SELECT * FROM student_details WHERE id = %s', [id])
    data = cur.fetchall()#all rows of table are fetched and returned as list of tuples 
    cur.close()
   
    return render_template('edit.html', student = data[0])

#update an row in the table
@app.route('/update/<id>', methods=['GET','POST'])
def update_student(id):
    if request.method == 'POST':
        fname = request.form['fname']
        lname = request.form['lname']
        email = request.form['email']

        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor) 

        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor) 
            cur.execute("""
            UPDATE student_details
            SET fname = %s,
            lname = %s,
            email = %s
            WHERE id = %s""", (fname, lname, email, id))

            
        except psycopg2.IntegrityError:
            conn.rollback()
            s = "SELECT * FROM student_details"
            cur.execute(s) # Execute the SQL
            list_users = cur.fetchall()#fetches a
            conn.commit()
            print(list_users)
            
            
            return render_template('index.html', list_users = list_users,msg='A user exists with same email try with new email')
    return redirect(url_for('Index'))
    
       
              
 
#To delete the row from table
@app.route('/delete/<string:id>', methods = ['POST','GET'])
def delete_student(id):
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
   
    cur.execute('DELETE FROM student_details WHERE id = {0}'.format(id))
    conn.commit()
    flash('Student Removed Successfully')
    return redirect(url_for('Index')) 

app.secret_key = 'the random string' 
if __name__ == "__main__":
   
    app.run(debug=True,port=5900)
   