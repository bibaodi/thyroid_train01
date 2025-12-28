import yaml
with open('requirements.txt') as f:
    deps = [l.strip() for l in f if l.strip() and not l.startswith('#')]
env = {'name': 'myenv', 'channels': ['defaults'], 'dependencies': ['pip'] + [{'pip': deps}]}
with open('environment.yml', 'w') as f:
    yaml.dump(env, f, default_flow_style=False)
