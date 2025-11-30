import uvicorn
import sys
from app.domain.model.svd import SVD_Predictor
from app.domain.model.slim import SLIM_Predictor
from app.domain.model.als import ALS_Predictor
from app.domain.model.ease import EASE_Predictor
from app.domain.model.hybrid import Hybrid_Predictor
from app.domain.model.knn import ItemKNN_Predictor

setattr(sys.modules['__main__'], 'EASE_Predictor', EASE_Predictor)
setattr(sys.modules['__main__'], 'ALS_Predictor', ALS_Predictor)
setattr(sys.modules['__main__'], 'SLIM_Predictor', SLIM_Predictor)
setattr(sys.modules['__main__'], 'SVD_Predictor', SVD_Predictor)
setattr(sys.modules['__main__'], 'Hybrid_Predictor', Hybrid_Predictor)
setattr(sys.modules['__main__'], 'ItemKNN_Predictor', ItemKNN_Predictor)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)