import os

def tree(path, prefix='', max_depth=4, depth=0):
    if depth > max_depth:
        return
    try:
        items = sorted([item for item in os.listdir(path) if not item.startswith('.')])
        dirs = [item for item in items if os.path.isdir(os.path.join(path, item))]
        files = [item for item in items if os.path.isfile(os.path.join(path, item))]
        
        for i, d in enumerate(dirs):
            is_last = i == len(dirs) - 1 and len(files) == 0
            print(f'{prefix}{"└──" if is_last else "├──"} {d}/')
            tree(os.path.join(path, d), prefix + ('    ' if is_last else '│   '), max_depth, depth + 1)
        
        for i, f in enumerate(files):
            is_last = i == len(files) - 1
            print(f'{prefix}{"└──" if is_last else "├──"} {f}')
    except PermissionError:
        print(f'{prefix}[Permission Denied]')
    except Exception as e:
        print(f'{prefix}[Error: {e}]')

root = r'c:\Repo\Crowd and Anomaly Detection\data'
if os.path.exists(root):
    print('data/')
    tree(root, max_depth=3)
else:
    print('Data directory not found')
