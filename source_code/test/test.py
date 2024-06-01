import os
def testfunc():
    print("just test a mode")
    print(os.path.dirname(os.path.abspath(__file__)))

def showcwd():
    print(os.getcwd())