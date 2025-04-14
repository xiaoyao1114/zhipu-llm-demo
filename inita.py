import pkg_resources

def generate_requirements(output_file='requirements.txt'):
    # 获取当前环境中的所有包及其版本
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    # 将包名和版本号写入到requirements.txt文件中
    with open(output_file, 'w') as f:
        for pkg_name, version in installed_packages.items():
            f.write(f'{pkg_name}=={version}\n')

# 调用函数生成requirements.txt文件
generate_requirements()