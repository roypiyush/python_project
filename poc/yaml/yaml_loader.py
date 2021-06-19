import yaml
from jinja2 import Environment, FileSystemLoader

sample_yaml = "sample.yaml"
with open(sample_yaml) as simple:
    simple = yaml.load(simple, Loader=yaml.FullLoader)
    print(simple)
    env = Environment(loader=FileSystemLoader('./'), trim_blocks=True, lstrip_blocks=True)
    tmpl = env.get_template(sample_yaml)
    d = dict(
         d1='1.2.3.4',
         username=u"Username",
         password=u"Password")
    print(tmpl.render(d))

