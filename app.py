from flask import Flask, render_template, request, jsonify
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json

app = Flask(__name__)

class MedicalTimelineExtractor:
    def __init__(self, reference_date: datetime = None):
        self.reference_date = reference_date or datetime.now()
        
        # Medical event keywords for categorization
        self.medical_keywords = {
            'Disease Onset/Progression': [
                'symptoms', 'began', 'started', 'onset', 'developed', 'appeared', 
                'worsened', 'improved', 'subsided', 'resolved', 'fever', 'cough',
                'pain', 'headache', 'nausea', 'fatigue', 'weakness', 'dizzy'
            ],
            'Treatment': [
                'medication', 'prescribed', 'started', 'completed', 'treatment',
                'therapy', 'surgery', 'procedure', 'dose', 'mg', 'administered',
                'antibiotic', 'medicine', 'drug', 'injection', 'infusion'
            ],
            'Appointment/Scheduling': [
                'appointment', 'visit', 'scheduled', 'follow-up', 'consultation',
                'test', 'exam', 'x-ray', 'scan', 'lab', 'blood work', 'results'
            ],
            'Other Clinical Observations': [
                'observed', 'noted', 'reports', 'presents', 'examination',
                'assessment', 'diagnosis', 'condition', 'patient', 'clinical'
            ]
        }
    
    def extract_dates_and_events(self, text: str) -> List[Dict]:
        """Extract temporal information and associated events from medical text"""
        events = []
        
        # Split text into sentences for better processing
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Extract dates from sentence
            extracted_dates = self._extract_dates_from_sentence(sentence)
            
            if extracted_dates:
                for date_info in extracted_dates:
                    event = {
                        'original_text': sentence,
                        'date': date_info['normalized_date'].isoformat() if date_info['normalized_date'] else None,
                        'date_text': date_info['original_text'],
                        'event_description': self._clean_event_description(sentence, date_info['original_text']),
                        'category': self._categorize_event(sentence),
                        'confidence': date_info.get('confidence', 'medium')
                    }
                    events.append(event)
            else:
                # If no explicit date, try to infer from context or skip
                if self._has_medical_content(sentence):
                    event = {
                        'original_text': sentence,
                        'date': None,
                        'date_text': 'Date not specified',
                        'event_description': sentence,
                        'category': self._categorize_event(sentence),
                        'confidence': 'low'
                    }
                    events.append(event)
        
        return events
    
    def _extract_dates_from_sentence(self, sentence: str) -> List[Dict]:
        """Extract and normalize dates from a sentence"""
        dates = []
        
        # Pattern for absolute dates - Updated to handle more formats
        absolute_patterns = [
            # Full month names with ordinals: "30th August 2025", "15th of August 2025"
            r'\b\d{1,2}(?:st|nd|rd|th)\s+(?:of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
            # Month names without ordinals: "August 30, 2025", "August 30 2025"
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*|\s+)\d{4}\b',
            # Abbreviated months: "Aug 30, 2025", "30 Aug 2025"
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:,\s*|\s+)\d{4}\b',
            r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b',
            # Numeric formats: "08/22/2025", "2025-08-22"
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            # Simple words
            r'\btoday\b',
            r'\byesterday\b',
            r'\btomorrow\b'
        ]
        
        # Pattern for relative dates
        relative_patterns = [
            r'\b(?:three|two|one|\d+)\s+(?:weeks?|days?|months?|years?)\s+ago\b',
            r'\b(?:three|two|one|\d+)\s+(?:weeks?|days?|months?|years?)\s+(?:prior|before|earlier)\b',
            r'\b(?:in|after)\s+(?:three|two|one|\d+)\s+(?:weeks?|days?|months?|years?)\b',
            r'\b(?:three|two|one|\d+)\s+(?:weeks?|days?|months?|years?)\s+(?:from now|later)\b',
            r'\bafter\s+\d+\s+days?\b',
            r'\b(?:last|next)\s+(?:week|month|year)\b',
            r'\bfor\s+the\s+past\s+(?:three|two|one|\d+)\s+(?:weeks?|days?|months?|years?)\b'
        ]
        
        all_patterns = absolute_patterns + relative_patterns
        
        for pattern in all_patterns:
            matches = re.finditer(pattern, sentence, re.IGNORECASE)
            for match in matches:
                date_text = match.group()
                normalized_date = self._normalize_date(date_text)
                if normalized_date:
                    dates.append({
                        'original_text': date_text,
                        'normalized_date': normalized_date,
                        'confidence': 'high' if pattern in absolute_patterns else 'medium'
                    })
        
        return dates
    
    def _simple_date_parse(self, date_text: str) -> Optional[datetime]:
        """Simple date parsing without external libraries"""
        try:
            # Remove any extra spaces
            date_text = re.sub(r'\s+', ' ', date_text.strip())
            
            # Month name mappings
            month_names = {
                'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
                'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6,
                'jul': 7, 'july': 7, 'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
                'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12
            }
            
            # Try ordinal formats: "30th August 2025", "15th of August 2025"
            match = re.match(r'(\d{1,2})(?:st|nd|rd|th)\s+(?:of\s+)?([a-zA-Z]+)\s+(\d{4})', date_text, re.IGNORECASE)
            if match:
                day, month_str, year = match.groups()
                month_num = month_names.get(month_str.lower())
                if month_num:
                    try:
                        return datetime(int(year), month_num, int(day)).date()
                    except ValueError:
                        return None
            
            # Try standard month name format: "August 30, 2025", "August 30 2025"
            match = re.match(r'([a-zA-Z]+)\s+(\d{1,2})(?:,\s*|\s+)(\d{4})', date_text, re.IGNORECASE)
            if match:
                month_str, day, year = match.groups()
                month_num = month_names.get(month_str.lower())
                if month_num:
                    try:
                        return datetime(int(year), month_num, int(day)).date()
                    except ValueError:
                        return None
            
            # Try reverse format: "30 Aug 2025", "30 August 2025"
            match = re.match(r'(\d{1,2})(?:st|nd|rd|th)?\s+([a-zA-Z]+)\s+(\d{4})', date_text, re.IGNORECASE)
            if match:
                day, month_str, year = match.groups()
                month_num = month_names.get(month_str.lower())
                if month_num:
                    try:
                        return datetime(int(year), month_num, int(day)).date()
                    except ValueError:
                        return None
            
            # Try MM/DD/YYYY or MM-DD-YYYY
            match = re.match(r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})', date_text)
            if match:
                month, day, year = match.groups()
                year = int(year)
                if year < 100:  # Handle 2-digit years
                    year = 2000 + year if year < 50 else 1900 + year
                try:
                    return datetime(year, int(month), int(day)).date()
                except ValueError:
                    return None
            
            # Try YYYY/MM/DD or YYYY-MM-DD
            match = re.match(r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', date_text)
            if match:
                year, month, day = match.groups()
                try:
                    return datetime(int(year), int(month), int(day)).date()
                except ValueError:
                    return None
            
            return None
        except Exception:
            return None

    def _normalize_date(self, date_text: str) -> Optional[datetime]:
        """Convert various date formats to normalized datetime"""
        try:
            # Handle common relative expressions manually
            date_text_lower = date_text.lower()
            
            if 'today' in date_text_lower:
                return self.reference_date.date()
            elif 'yesterday' in date_text_lower:
                return (self.reference_date - timedelta(days=1)).date()
            elif 'tomorrow' in date_text_lower:
                return (self.reference_date + timedelta(days=1)).date()
            
            # Handle "for the past X weeks/days/months"
            if 'for the past' in date_text_lower:
                return self._parse_past_duration(date_text)
            
            # Handle relative time expressions
            if 'ago' in date_text_lower or 'prior' in date_text_lower or 'before' in date_text_lower or 'earlier' in date_text_lower:
                return self._parse_relative_past(date_text)
            elif 'from now' in date_text_lower or 'later' in date_text_lower or date_text_lower.startswith('in ') or date_text_lower.startswith('after '):
                return self._parse_relative_future(date_text)
            elif 'last' in date_text_lower:
                return self._parse_last_time_reference(date_text)
            elif 'next' in date_text_lower:
                return self._parse_next_time_reference(date_text)
            
            # Try simple date parsing for absolute dates
            return self._simple_date_parse(date_text)
            
        except Exception:
            return None

    def _parse_past_duration(self, date_text: str) -> Optional[datetime]:
        """Parse 'for the past X weeks' type expressions"""
        try:
            match = re.search(r'for\s+the\s+past\s+(three|two|one|\d+)\s+(weeks?|days?|months?|years?)', date_text.lower())
            if not match:
                return None
            
            number_text, unit = match.groups()
            number_map = {'one': 1, 'two': 2, 'three': 3}
            number = number_map.get(number_text, int(number_text) if number_text.isdigit() else 1)
            
            # Return the start date of the duration
            if 'day' in unit:
                return (self.reference_date - timedelta(days=number)).date()
            elif 'week' in unit:
                return (self.reference_date - timedelta(weeks=number)).date()
            elif 'month' in unit:
                return self._add_months(self.reference_date.date(), -number)
            elif 'year' in unit:
                return self._add_months(self.reference_date.date(), -number * 12)
            
            return None
        except Exception:
            return None
    
    def _add_months(self, date, months):
        """Add months to a date (simplified version)"""
        year = date.year
        month = date.month + months
        
        while month > 12:
            month -= 12
            year += 1
        while month < 1:
            month += 12
            year -= 1
            
        # Handle day overflow
        try:
            return datetime(year, month, date.day).date()
        except ValueError:
            # If day doesn't exist in target month, use last day of month
            import calendar
            last_day = calendar.monthrange(year, month)[1]
            return datetime(year, month, min(date.day, last_day)).date()

    def _parse_relative_past(self, date_text: str) -> Optional[datetime]:
        """Parse relative past dates like 'three weeks ago'"""
        try:
            # Extract number and unit
            match = re.search(r'(three|two|one|\d+)\s+(weeks?|days?|months?|years?)', date_text.lower())
            if not match:
                return None
            
            number_text, unit = match.groups()
            
            # Convert word numbers to integers
            number_map = {'one': 1, 'two': 2, 'three': 3}
            number = number_map.get(number_text, int(number_text) if number_text.isdigit() else 1)
            
            if 'day' in unit:
                return (self.reference_date - timedelta(days=number)).date()
            elif 'week' in unit:
                return (self.reference_date - timedelta(weeks=number)).date()
            elif 'month' in unit:
                return self._add_months(self.reference_date.date(), -number)
            elif 'year' in unit:
                return self._add_months(self.reference_date.date(), -number * 12)
            
            return None
        except Exception:
            return None
    
    def _parse_relative_future(self, date_text: str) -> Optional[datetime]:
        """Parse relative future dates like 'in two weeks'"""
        try:
            match = re.search(r'(three|two|one|\d+)\s+(weeks?|days?|months?|years?)', date_text.lower())
            if not match:
                return None
            
            number_text, unit = match.groups()
            number_map = {'one': 1, 'two': 2, 'three': 3}
            number = number_map.get(number_text, int(number_text) if number_text.isdigit() else 1)
            
            if 'day' in unit:
                return (self.reference_date + timedelta(days=number)).date()
            elif 'week' in unit:
                return (self.reference_date + timedelta(weeks=number)).date()
            elif 'month' in unit:
                return self._add_months(self.reference_date.date(), number)
            elif 'year' in unit:
                return self._add_months(self.reference_date.date(), number * 12)
            
            return None
        except Exception:
            return None
    
    def _parse_last_time_reference(self, date_text: str) -> Optional[datetime]:
        """Parse 'last week/month/year' references"""
        try:
            if 'week' in date_text.lower():
                return (self.reference_date - timedelta(weeks=1)).date()
            elif 'month' in date_text.lower():
                return self._add_months(self.reference_date.date(), -1)
            elif 'year' in date_text.lower():
                return self._add_months(self.reference_date.date(), -12)
            return None
        except Exception:
            return None
    
    def _parse_next_time_reference(self, date_text: str) -> Optional[datetime]:
        """Parse 'next week/month/year' references"""
        try:
            if 'week' in date_text.lower():
                return (self.reference_date + timedelta(weeks=1)).date()
            elif 'month' in date_text.lower():
                return self._add_months(self.reference_date.date(), 1)
            elif 'year' in date_text.lower():
                return self._add_months(self.reference_date.date(), 12)
            return None
        except Exception:
            return None
    
    def _categorize_event(self, text: str) -> str:
        """Categorize the medical event based on keywords"""
        text_lower = text.lower()
        
        category_scores = {}
        for category, keywords in self.medical_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            category_scores[category] = score
        
        if not any(category_scores.values()):
            return "Other Clinical Observations"
        
        return max(category_scores.items(), key=lambda x: x[1])[0]
    
    def _clean_event_description(self, sentence: str, date_text: str) -> str:
        """Clean the event description by removing redundant date information"""
        cleaned = sentence.replace(date_text, '').strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Remove extra whitespace
        return cleaned if cleaned else sentence
    
    def _has_medical_content(self, sentence: str) -> bool:
        """Check if sentence contains medical content worth including"""
        medical_indicators = [
            'patient', 'symptoms', 'diagnosis', 'treatment', 'medication',
            'test', 'exam', 'results', 'appointment', 'visit', 'prescribed'
        ]
        return any(indicator in sentence.lower() for indicator in medical_indicators)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_medical_text():
    try:
        data = request.json
        medical_text = data.get('medical_text', '')
        reference_date_str = data.get('reference_date', datetime.now().isoformat())
        
        if not medical_text.strip():
            return jsonify({'error': 'Please provide medical text to analyze'}), 400
        
        # Parse reference date
        try:
            reference_date = datetime.fromisoformat(reference_date_str.replace('Z', '+00:00'))
        except:
            reference_date = datetime.now()
        
        # Create extractor and process text
        extractor = MedicalTimelineExtractor(reference_date=reference_date)
        events = extractor.extract_dates_and_events(medical_text)
        
        # Sort events by date
        dated_events = [e for e in events if e['date'] is not None]
        undated_events = [e for e in events if e['date'] is None]
        
        # Sort dated events chronologically
        dated_events.sort(key=lambda x: x['date'])
        
        return jsonify({
            'success': True,
            'events': dated_events + undated_events,
            'dated_events': len(dated_events),
            'undated_events': len(undated_events),
            'total_events': len(events)
        })
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/sample')
def get_sample_text():
    sample_text = """Patient Name: Jane Smith
Date of Visit: August 22, 2025

Patient presents with a persistent cough and fever. She reports that the symptoms began approximately three weeks ago. The fever subsided after 5 days, but the cough has remained.

We started her on a course of Azithromycin 250mg for 5 days, which she completed yesterday.

A chest X-ray was performed today. Results are pending. We have scheduled a follow-up appointment for two weeks from now to discuss the results and next steps.

The patient also mentioned experiencing fatigue that started last month. Blood work was ordered and completed on August 20, 2025. Results showed slightly elevated white blood cell count.

Follow-up scheduled for September 5, 2025."""
    
    return jsonify({'sample_text': sample_text})

if __name__ == '__main__':
    app.run(debug=True)