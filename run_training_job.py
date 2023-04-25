from datajoint_tutorial.schema import Train

Train().populate(reserve_jobs=True, order="random", suppress_errors=True)