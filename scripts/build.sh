python -m setuptools_git_versioning

version="0.0.19"

git add .
git commit -m "version v${version}"

git tag -a v${version} -m "v${version}"

python -m build

twine upload --repository testpypi dist/chromatinhd-${version}.tar.gz --verbose

git push --tags

gh release create v${version} -t "v${version}" -n "v${version}" dist/chromatinhd-${version}.tar.gz

twine upload dist/chromatinhd-${version}.tar.gz --verbose