import bw2data as bd
import bw2io as bi

def init_project():
    if "ecoinvent-3.10-cutoff" in bd.databases:
        print("ecoinvent 3.10 is already present in the project")
    else:
        print("Importing ecoinvent 3.10")
        bi.import_ecoinvent_release(
            version="3.10",
            system_model="cutoff",
            username="EcoInventQSA2",
            password="EcoinventDTUQSA22",
        )
        
if __name__ == "__main__":
    if "default" not in bd.projects:
        bd.projects.create_project("default")
    else:
        bd.projects.set_current("default")
    init_project()