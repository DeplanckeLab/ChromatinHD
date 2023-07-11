import diskcache
import appdirs

cache = diskcache.Cache(appdirs.user_cache_dir('biomart'))
