#!/bin/bash
# DocuMind Project Runner for Linux/Mac
# Convenient script to run various project commands

show_menu() {
    clear
    echo "========================================"
    echo "     DocuMind - Document Automation"
    echo "========================================"
    echo ""
    echo "1. Run Demo"
    echo "2. Start API Server"
    echo "3. Start Web UI"
    echo "4. Start Both (API + UI)"
    echo "5. Run Tests"
    echo "6. Check System Health"
    echo "7. View Logs"
    echo "8. Clear Data"
    echo "9. Docker - Build and Run"
    echo "0. Exit"
    echo ""
    read -p "Enter your choice (0-9): " choice
}

activate_venv() {
    source .venv/bin/activate
}

while true; do
    show_menu
    case $choice in
        1)
            echo ""
            echo "Running demo..."
            echo ""
            activate_venv
            python demo.py
            read -p "Press Enter to continue..."
            ;;
        2)
            echo ""
            echo "Starting API server on http://localhost:8000"
            echo "Press Ctrl+C to stop"
            echo ""
            activate_venv
            uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
            ;;
        3)
            echo ""
            echo "Starting Web UI on http://localhost:8501"
            echo "Press Ctrl+C to stop"
            echo ""
            activate_venv
            streamlit run app/ui.py
            ;;
        4)
            echo ""
            echo "Starting API and UI..."
            echo ""
            echo "Opening API in background..."
            activate_venv
            uvicorn app.main:app --reload > api.log 2>&1 &
            API_PID=$!
            sleep 5
            echo "Opening UI in background..."
            streamlit run app/ui.py > ui.log 2>&1 &
            UI_PID=$!
            echo ""
            echo "Both services started!"
            echo "API: http://localhost:8000 (PID: $API_PID)"
            echo "UI: http://localhost:8501 (PID: $UI_PID)"
            echo ""
            read -p "Press Enter to stop both services..."
            kill $API_PID $UI_PID 2>/dev/null
            ;;
        5)
            echo ""
            echo "Running tests..."
            echo ""
            activate_venv
            pytest tests/ -v
            read -p "Press Enter to continue..."
            ;;
        6)
            echo ""
            echo "Checking system health..."
            echo ""
            curl -s http://localhost:8000/health | python -m json.tool
            if [ $? -ne 0 ]; then
                echo ""
                echo "ERROR: API server is not running!"
                echo "Please start the API server first (option 2)"
            fi
            echo ""
            read -p "Press Enter to continue..."
            ;;
        7)
            echo ""
            echo "Viewing recent logs..."
            echo ""
            if [ -f documind.log ]; then
                tail -n 50 documind.log
            else
                echo "No log file found. Run the application first."
            fi
            echo ""
            read -p "Press Enter to continue..."
            ;;
        8)
            echo ""
            echo "WARNING: This will delete all indexed documents and outputs!"
            read -p "Are you sure? (yes/no): " confirm
            if [ "$confirm" = "yes" ]; then
                echo "Clearing data..."
                rm -rf data/faiss_index
                rm -rf data/document_store
                rm -f data/outputs/*
                echo "Data cleared!"
            else
                echo "Cancelled."
            fi
            echo ""
            read -p "Press Enter to continue..."
            ;;
        9)
            echo ""
            echo "Building and running Docker containers..."
            echo ""
            docker-compose build
            docker-compose up -d
            echo ""
            echo "Containers started!"
            echo "API: http://localhost:8000"
            echo "UI: http://localhost:8501"
            echo ""
            echo "To stop: docker-compose down"
            echo ""
            read -p "Press Enter to continue..."
            ;;
        0)
            echo ""
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid choice!"
            sleep 2
            ;;
    esac
done
