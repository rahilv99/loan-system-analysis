from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from .table_parser import TableParser
from .standardize import process_bank_statements
from .statement_insights import StatementInsights
import os
import pickle as pkl


@csrf_exempt
def upload_pdf(request):
    if request.method == 'POST' and request.FILES['pdf']:
        pdf_file = request.FILES['pdf']
        file_path = default_storage.save(pdf_file.name, pdf_file)

        if (True):
            # Process the PDF
            parser = TableParser()
            tables = parser.parse_pdf_tables(file_path)
        else:
            file_path = 'statement_2.pkl'
            with open(file_path, 'rb') as f:
                tables = pkl.load(f)

        results = process_bank_statements(tables)
        statement_analyzer = StatementInsights()
        insights = statement_analyzer.execute(results[0])

        # Clean up the uploaded file
        os.remove(file_path)

        return JsonResponse({'insights': insights})
    return JsonResponse({'error': 'Invalid request'}, status=400)
