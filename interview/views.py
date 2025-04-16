import os
import traceback
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render

from .analyzer.facial import analyze_facial_emotions
from .analyzer.voice import extract_audio, analyze_voice_confidence
from .analyzer.transcript import transcribe_audio, analyze_technical_content

def homepage(request):
    return render(request, 'homepage.html')

def interview_page(request):
    return render(request, 'interview.html')

def result_page(request):
    facial = request.session.get('facial_confidence', 0)
    voice = request.session.get('voice_confidence', 0)
    technical = request.session.get('technical_score', 0)
    transcript = request.session.get('transcript', 'Transcript not available.')

    # Final composite score calculation (e.g., 40% facial, 30% voice, 30% technical)
    final_score = round((facial * 0.4) + (voice * 0.3) + (technical * 0.3), 2)

    return render(request, 'result.html', {
        'facial': facial,
        'voice': voice,
        'technical': technical,
        'final_score': final_score,
        'transcript': transcript,
    })

@csrf_exempt
def upload_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        try:
            video = request.FILES['video']
            save_dir = os.path.join(settings.MEDIA_ROOT, 'interviews')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, video.name)
            
            # Save the uploaded video file
            with open(save_path, 'wb+') as f:
                for chunk in video.chunks():
                    f.write(chunk)
                    
            # Step 1: Perform facial emotion analysis
            try:
                facial_score = analyze_facial_emotions(save_path)
            except Exception as e:
                facial_score = 0
                print("Error during facial analysis:")
                traceback.print_exc()
            
            # Step 2: Extract audio from the video file
            try:
                audio_path = save_path.rsplit('.', 1)[0] + '.wav'
                extract_audio(save_path, audio_path)
            except Exception as e:
                print("Error during audio extraction:")
                traceback.print_exc()
                return JsonResponse({'error': 'Audio extraction failed.'}, status=500)
            
            # Step 3: Analyze voice features to compute voice confidence
            try:
                voice_score = analyze_voice_confidence(audio_path)
            except Exception as e:
                voice_score = 0
                print("Error during voice analysis:")
                traceback.print_exc()
            
            # Step 4: Transcribe the extracted audio
            try:
                transcript = transcribe_audio(audio_path)
            except Exception as e:
                transcript = "Transcription failed."
                print("Error during transcription:")
                traceback.print_exc()
            
            # Step 5: Analyze transcript for technical content and language quality
            try:
                technical_score = analyze_technical_content(transcript)
            except Exception as e:
                technical_score = 0
                print("Error during technical content analysis:")
                traceback.print_exc()
            
            # Save the computed scores and transcript into the session for later display
            request.session['facial_confidence'] = float(facial_score)
            request.session['voice_confidence'] = float(voice_score)
            request.session['technical_score'] = float(technical_score)
            request.session['transcript'] = transcript

            return JsonResponse({'message': 'Video uploaded and processed successfully!'})
        except Exception as e:
            print("General error in upload_video:")
            traceback.print_exc()
            return JsonResponse({'error': 'Internal server error during video processing.'}, status=500)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)