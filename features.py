from pydantic import BaseModel


class Features(BaseModel):
    sex: int
    age: int
    anatom_site: str

def is_valid(anatom_site):
        location = ['head/neck' ,'upper extremity', 'lower extremity', 'torso' , 'palms/soles','oral/genital']
        if anatom_site in location:
            return True     
        else:
            return False
    # values for anatom_site: 
    # { 'head/neck' ,
    # 'upper extremity',
    #  'lower extremity', 
    # 'torso' , 
    # 'palms/soles',
    # 'oral/genital'
    # }
