-- product 임베딩 컬럼 분리.
--
-- 백엔드 마이그레이션(V2__initial_schema.sql:52)은 `embedding vector(768)` 한 컬럼이다.
-- 우리는 텍스트 1024차원(BGE-M3) · 이미지 768차원(SigLIP)을 따로 쓰므로 자리가 맞지 않는다.
-- 하나로 합치지 않는 이유는 차원도 축의 의미도 다르기 때문이다.
--
-- **`embedding` 은 드롭하지 않는다.** 백엔드 `Product.java:51` 이 이 컬럼을 매핑하고
-- `ddl-auto: validate` 라, 지우면 백엔드가 부팅에서 죽는다. 우리는 안 쓰고 두면 된다.
--
-- 로컬 DB 를 다시 만든 뒤(백엔드가 Flyway 로 스키마·시드를 생성한다) 실행한다.
--
--     docker exec -i dib-postgres psql -U postgres -d dib < scripts/split_embedding_columns.sql
--
-- 여러 번 실행해도 안전하다.

BEGIN;

ALTER TABLE product ADD COLUMN IF NOT EXISTS text_embedding  vector(1024);
ALTER TABLE product ADD COLUMN IF NOT EXISTS image_embedding vector(768);

-- 임베딩 배치(scripts/embed_products.py)가 쓰는 유일한 권한이다.
-- 계정을 나눠 쓸 때만 필요하고, postgres 로 붙는 로컬에서는 없어도 된다.
--
--   GRANT UPDATE (text_embedding, image_embedding) ON product TO dib_ai;

COMMIT;

-- 확인
--   \d product
--   SELECT count(*) FILTER (WHERE text_embedding IS NOT NULL) FROM product;
