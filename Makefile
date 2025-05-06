install_backend:
	cd backend && chmod +x ./scripts/install.sh
	cd backend && ./scripts/install.sh
	cd backend && source .venv/bin/activate; python -m spacy download en_core_web_lg

install_frontend:
	cd frontend && pnpm install

lint_frontend:
	cd frontend && pnpm run lint

format_frontend:
	cd frontend && pnpm run format

start_frontend:
	cd frontend && pnpm run dev

start_backend:
	cd backend && source .venv/bin/activate; python src/main.py

install: install_backend install_frontend

lint: lint_frontend

format: format_frontend

clean:
	find backend/src -type d -name "__pycache__" -exec rm -rf {} \; || true
	rm -rf backend/src/data

# Database and ChromaDB server management commands
start_db:
	docker compose up -d chroma

# Wait for services to be ready
wait_db:
	@echo "Waiting for ChromaDB server to be ready..."
	sleep 5 # Give some time for ChromaDB server to initialize
	curl -s http://localhost:8010/api/v2/heartbeat || echo "ChromaDB server not responding yet"

# Stop all services
stop_db:
	docker compose down

# Check status of all services
db_status:
	docker compose ps

# Check Docker volume sizes for ChromaDB
volume_sizes:
	@echo "\nDocker volume sizes:\n"
	@docker system df -v | grep -A 10 "VOLUME NAME"
	@echo "\nDetailed volume information:\n"
	@echo "ChromaDB volume:" && docker volume inspect large-document-rag_chroma_data | grep Mountpoint

# Reset vector database (warning: this will delete all data)
reset_db: stop_db
	docker volume rm large-document-rag_chroma_data || true
	docker compose up -d chroma

# Completely destroy all database data (use with caution)
destroy_db:
	@echo "WARNING: This will destroy all data in ChromaDB. Are you sure? (y/n)" && read ans && [ $${ans:-N} = y ]
	docker compose down --volumes --remove-orphans
	docker volume rm large-document-rag_chroma_data || true
	rm -rf backend/src/data || true
	@echo "All vector database data destroyed."

# Usage: make explore_chroma COLLECTION=collection_name
explore_chroma:
	cd backend && source .venv/bin/activate && python tools/explore_chroma.py $(COLLECTION)

# Financial document validation scripts

# Usage: make validate_financial_chunks COLLECTION=collection_name [TESTS=--all|--queries|--boundaries|--entities|--embeddings]
validate_financial_chunks:
	cd backend && source .venv/bin/activate; python tools/validate_financial_chunks.py

test_financial_chunking:
	cd backend && source .venv/bin/activate; python tools/test_financial_chunking.py

# Run all financial validations with default collection
validate_financial: 
	@echo "Running financial chunking validation..."
	make validate_financial_chunks COLLECTION=$(COLLECTION) TESTS=--all