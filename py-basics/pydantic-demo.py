from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str='kash' #default value
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt = 0, lt = 10, default = 5, description = 'description of this field')

new_student = {'age': '342', 'email': 'abs@gmail.com'}

student = Student(**new_student)

student_dict = dict(student)
print(student_dict['name'])

student_json = student.model_dump_json()
print(student_json)

print(student)