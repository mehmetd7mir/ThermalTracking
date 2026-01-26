
import yaml
import argparse
import logging
import csv
from pathlib import Path
from collections import Counter
import sys
import shutil  # Dosya kopyalama/taşıma için
import time
# === 0. KONFİGÜRASYON ve GLOBAL İŞLEMLER ===

STANDARD_CLASSES = {
    'bird': 0,
    'drone': 1,
    'helicopter': 2,
    'plane': 3
}
STANDARD_ID_TO_NAME = {v: k for k, v in STANDARD_CLASSES.items()}

VARIATION_MAP = {

    'Airplane': 'plane',
    'AIRPLANE': 'plane',
    'airplane': 'plane',
    'air plane': 'plane',
    'aeroplane': 'plane',
    'Aeroplane': 'plane',
    'AEROPLANE': 'plane',
    'plane': 'plane',
    'Plane': 'plane',
    'PLANE': 'plane',

    'drone': 'drone',
    'Drone': 'drone',
    'DRONE': 'drone',
    'uav': 'drone',
    'Uav': 'drone',
    'UAV': 'drone',

    'helicopter': 'helicopter',
    'Helicopter': 'helicopter',
    'HELICOPTER': 'helicopter',
    'helocopter': 'helicopter',
    'Helocopter': 'helicopter',

    'bird': 'bird',
    'Bird': 'bird',
    'BIRD': 'bird',

}

DEFAULT_DATA_DIR = '../MegaSet_RawData'
DEFAULT_OUTPUT_DIR = '../MegaSet_Cleaned'
DEFAULT_REPORT_FILE = 'scan_report.csv'

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


# ----------------------------------------------------------------------------
# BÖLÜM 1: DatasetAnalyzer Class (Veri Seti YAML Analizi ve Eşleştirme)
# ----------------------------------------------------------------------------
class DatasetAnalyzer:
    """
    Tek bir veri setinin data.yaml dosyasını okur, geçerliliğini kontrol eder
    ve etiketleri standart ID'lere eşleştirir.
    """

    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_path.name
        self.yaml_data = None
        self.original_class_names = None  # {index: name}
        self.yaml_issues = {}  # Hata/uyarı bilgileri
        self.standardized_mapping = {}  # {orig_idx: std_idx}
        self.ready_for_txt = False
        self.report = {"dataset_folder": self.dataset_name}  # Rapor satırı için başlangıç

    def _read_yaml(self):
        """ data.yaml dosyasını okur ve geçerliliğini kontrol eder. """
        yaml_path = self.dataset_path / 'data.yaml'
        self.report["yaml_path"] = str(yaml_path)
        if not yaml_path.is_file():
            logging.warning(f"  [{self.dataset_name}] data.yaml bulunamadı.")
            return False

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    logging.warning(f"  [{self.dataset_name}] data.yaml boş.")
                    return False
                f.seek(0)
                self.yaml_data = yaml.safe_load(f)
                return self.yaml_data is not None
        except yaml.YAMLError as e:
            logging.error(f"  [{self.dataset_name}] YAML ayrıştırma hatası: {e}")
        except Exception as e:
            logging.error(f"  [{self.dataset_name}] YAML okuma hatası: {e}")
        return False

    def _validate_yaml_names_section(self, yaml_data):
        """
        Yapı, boş isim, ardışık indeks ve çift isim kontrolü yapar.
        """
        issues = {
            "valid_structure": False, "empty_names_found": False,
            "non_contiguous_indices": False, "duplicate_original_names": [],
            "nc_value": None, "nc_mismatch": None
        }

        if 'names' not in yaml_data or not isinstance(yaml_data['names'], (list, dict)):
            return None, issues

        raw_names_data = yaml_data['names']
        temp_names = {}

        try:
            if isinstance(raw_names_data, list):
                temp_names = {i: str(name).strip() for i, name in enumerate(raw_names_data)}
            else:
                for k, v in raw_names_data.items():
                    try:
                        key_int = int(str(k).strip())
                        if key_int < 0: continue
                        temp_names[key_int] = str(v).strip()
                    except ValueError:
                        continue

            issues["valid_structure"] = True

        except Exception as e:
            logging.error(f"  [{self.dataset_name}][YAML] İsimler bölümü işlenirken hata: {e}")
            return None, issues

        original_class_names = temp_names

        if not original_class_names:
            issues["empty_names_found"] = True
            return None, issues

        # Kontroller
        if "" in original_class_names.values():
            issues["empty_names_found"] = True

        indices = sorted(original_class_names.keys())
        expected_indices = list(range(len(indices)))
        if indices != expected_indices:
            issues["non_contiguous_indices"] = True

        name_counts = Counter(original_class_names.values())
        duplicates = [name for name, count in name_counts.items() if count > 1 and name != ""]
        if duplicates:
            issues["duplicate_original_names"] = duplicates

        # nc kontrolü
        if 'nc' in yaml_data:
            try:
                nc_value = int(yaml_data['nc'])
                issues["nc_value"] = nc_value
                if nc_value != len(original_class_names):
                    issues["nc_mismatch"] = True
            except:
                issues["nc_mismatch"] = True

        return original_class_names, issues

    def validate_and_map(self):
        """ Ana doğrulama ve eşleştirme sürecini yürütür. """
        self.report["yaml_ok"] = self._read_yaml()
        if not self.report["yaml_ok"]: return False

        self.original_class_names, self.yaml_issues = self._validate_yaml_names_section(self.yaml_data)
        self.report["original_class_count"] = len(self.original_class_names) if self.original_class_names else 0

        if not self.original_class_names: return False

        # --- TEST KLASÖRÜ VE NC DEĞERİ RAPORLAMASI ---
        test_path = self.dataset_path / 'test'
        self.report["test_missing"] = not test_path.is_dir()
        if self.report["test_missing"]:
            logging.warning(f"  [{self.dataset_name}] 'test' klasörü bulunamadı (Ham veri stratejisi).")

        self.report.update({
            "contiguous_indices_ok": not self.yaml_issues["non_contiguous_indices"],
            "duplicate_original_names": str(self.yaml_issues["duplicate_original_names"]),
            "empty_names_found": self.yaml_issues["empty_names_found"],
            "nc_mismatch": self.yaml_issues["nc_mismatch"],
            "nc_value": self.yaml_issues["nc_value"],
        })
        # ---------------------------------------------

        # 1. Standart Etiketlere Eşleştirme (İsim -> ID)
        unknown_names_found = []
        normalized_names_list = []
        temp_mapping = {}
        all_known_and_valid = True

        for original_idx, original_name in self.original_class_names.items():
            stripped_name = original_name.strip()

            standard_name = VARIATION_MAP.get(stripped_name)

            if standard_name:
                standard_idx = STANDARD_CLASSES.get(standard_name)
                temp_mapping[original_idx] = standard_idx
                normalized_names_list.append(standard_name)
            else:
                unknown_names_found.append(original_name)
                all_known_and_valid = False

        # 2. Eşleştirme sonrası çift isim kontrolü
        normalized_counts = Counter(normalized_names_list)
        duplicate_normalized_names = [name for name, count in normalized_counts.items() if count > 1]

        self.standardized_mapping = temp_mapping
        self.report["unknown_names_found"] = str(unknown_names_found)
        self.report["duplicate_normalized_names"] = str(duplicate_normalized_names)
        self.report["mapping_preview"] = self._get_mapping_preview()

        # 3. Nihai Hazırlık Kontrolü (TXT İşlemeye Geçiş)
        is_contiguous = not self.yaml_issues["non_contiguous_indices"]
        is_clean_yaml = not self.yaml_issues["duplicate_original_names"] and not self.yaml_issues["empty_names_found"]

        if is_contiguous and is_clean_yaml and all_known_and_valid:
            self.ready_for_txt = True
            logging.info(f"  [{self.dataset_name}] Doğrulama başarılı. TXT işleme hazır.")
        else:
            self.ready_for_txt = False
            logging.warning(f"  [{self.dataset_name}] TXT işleme atlandı. (Hata var, raporda detaylı kontrol et.)")

        self.report["ready_for_txt_processing"] = self.ready_for_txt
        return self.ready_for_txt

    def get_mapping(self) -> dict:
        return self.standardized_mapping

    def get_report_row(self) -> dict:
        return self.report

    def _get_mapping_preview(self):
        preview_items = []
        for orig_idx, std_idx in self.standardized_mapping.items():
            orig_name = self.original_class_names.get(orig_idx, '??')
            std_name = STANDARD_ID_TO_NAME.get(std_idx, '??')
            preview_items.append(f"{orig_idx}({orig_name})->{std_idx}({std_name})")
        return "; ".join(preview_items)

    def update_data_yaml(self, output_dataset_path: Path, dry_run: bool):
        """Temizlenmiş data.yaml dosyası oluşturur"""
        if not self.ready_for_txt:
            logging.warning(f"  [{self.dataset_name}] YAML güncelleme atlandı (hazır değil)")
            return False
            
        # Standart sınıf isimlerini sıralı liste olarak oluştur
        standard_names = ['bird', 'plane', 'helicopter', 'drone']
        
        # Yeni data.yaml içeriği
        new_yaml_content = {
            'train': '../train/images',
            'val': '../valid/images',
            'test': '../test/images',
            'nc': 4,
            'names': standard_names
        }
        
        # Roboflow bilgilerini koru (varsa)
        if self.yaml_data and 'roboflow' in self.yaml_data:
            new_yaml_content['roboflow'] = self.yaml_data['roboflow']
        
        output_yaml_path = output_dataset_path / 'data.yaml'
        
        if dry_run:
            logging.info(f"  [{self.dataset_name}] DRY RUN: YAML güncellenecek -> {output_yaml_path}")
            logging.info(f"  [{self.dataset_name}] Yeni içerik: {new_yaml_content}")
            return True
        
        try:
            # Çıktı klasörünü oluştur
            output_dataset_path.mkdir(parents=True, exist_ok=True)
            
            # YAML dosyasını yaz
            with open(output_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(new_yaml_content, f, default_flow_style=False, allow_unicode=True)
            
            logging.info(f"  [{self.dataset_name}] YAML güncellendi: {output_yaml_path}")
            return True
            
        except Exception as e:
            logging.error(f"  [{self.dataset_name}] YAML güncelleme hatası: {e}")
            return False


# ----------------------------------------------------------------------------
# BÖLÜM 2: LabelProcessor Class (TXT Dosya İşlemleri)
# ----------------------------------------------------------------------------
class LabelProcessor:
    """
    Belirli bir veri setinin etiket dosyalarını okur, sınıfları eşleştirir
    ve temizlenmiş dosyaları yazar.
    """

    def __init__(self, dataset_path: Path, output_dataset_path: Path,
                 mapping: dict, dry_run: bool, error_log_path: Path):
        self.dataset_path = dataset_path
        self.output_dataset_path = output_dataset_path
        self.mapping = mapping
        self.dry_run = dry_run
        self.stats = self._initialize_stats()
        self.dataset_name = dataset_path.name
        self.subdirs_to_check = ['train', 'valid', 'test']
        self.error_log_path = error_log_path

    def _initialize_stats(self):
        return {
            "txt_files_found": 0, "txt_files_processed": 0, "lines_processed": 0,
            "lines_modified": 0, "lines_skipped_unknown": 0, "lines_skipped_invalid_format": 0,
            "txt_errors": 0
        }

    def _log_unknown_label(self, label_file_name, line_num, original_class_id, max_retries=3):
        for attempt in range(max_retries):
            try:
                with open(self.error_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"[{self.dataset_name}] {label_file_name}:{line_num}: ID {original_class_id} bilinmiyor.\n")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.error(f"    [ERROR][LOG] {max_retries} denemeden sonra hata dosyasına yazılamadı: {e}")
                    self.stats["txt_errors"] += 1
                time.sleep(1)  # 1 saniye bekle ve tekrar dene

    def _create_output_dirs(self, subdir_name):
        output_labels_path = self.output_dataset_path / subdir_name / 'labels'
        output_images_path = self.output_dataset_path / subdir_name / 'images'

        try:
            output_labels_path.mkdir(parents=True, exist_ok=True)
            output_images_path.mkdir(parents=True, exist_ok=True)
            return output_labels_path, output_images_path
        except Exception as e:
            logging.error(f"    [ERROR][FS] Çıktı klasörü oluşturulamadı: {e}")
            self.stats["txt_errors"] += 1
            return None, None

    def _process_subdir(self, subdir_name):
        source_labels_path = self.dataset_path / subdir_name / 'labels'
        source_images_path = self.dataset_path / subdir_name / 'images'

        if not source_labels_path.is_dir():
            if (self.dataset_path / subdir_name).is_dir():
                logging.debug(f"    [TXT] Etiket klasörü yok: {subdir_name}. Atlanıyor.")
            return

        # DEBUG 1: Kaç tane .txt dosyası bulunduğunu listele
        try:
            found_files = list(source_labels_path.glob('*.txt'))
            logging.info(f"      [DEBUG] Found {len(found_files)} TXT files in {source_labels_path}")
        except Exception as e:
            logging.error(f"    [ERROR] Could not scan for TXT files in {source_labels_path}. Error: {e}")
            self.stats["txt_errors"] += 1
            return
        # DEBUG 1 Bitiş

        output_labels_path, output_images_path = None, None
        if not self.dry_run:
            output_labels_path, output_images_path = self._create_output_dirs(subdir_name)
            if not output_labels_path: return

        logging.info(f"      [TXT] {subdir_name} etiketleri taranıyor...")

        processed_files_in_subdir = 0

        for label_file in found_files:  # Listeyi kullan
            self.stats["txt_files_found"] += 1
            new_lines = []

            try:
                # 1. Dosyayı Oku ve İşle
                with open(label_file, 'r', encoding='utf-8') as infile:
                    # DEBUG 2: Dosyanın açıldığını doğrula
                    logging.debug(f"        [DEBUG] Opened {label_file.name}")
                    # DEBUG 2 Bitiş
                    for line_num, line in enumerate(infile):
                        self.stats["lines_processed"] += 1
                        line = line.strip()
                        if not line: continue

                        parts = line.split()

                        # 1.1 Format Kontrolü
                        if len(parts) != 5:
                            self.stats["lines_skipped_invalid_format"] += 1
                            continue

                        # 1.2 ID ve Koordinat Kontrolü
                        try:
                            original_class_id = int(parts[0])
                            coords = [float(p) for p in parts[1:]]
                            if not all(0.0 <= c <= 1.0 for c in coords):
                                self.stats["lines_skipped_invalid_format"] += 1
                                continue
                        except ValueError:
                            self.stats["lines_skipped_invalid_format"] += 1
                            continue

                        # 1.3 Harita Uygulama (Düzeltilmiş Mantık)
                        if original_class_id in self.mapping:
                            standard_class_id = self.mapping[original_class_id]
                            new_line = f"{standard_class_id} {' '.join(parts[1:])}"
                            new_lines.append(new_line)

                            if original_class_id != standard_class_id:
                                self.stats["lines_modified"] += 1
                        else:
                            # Bilinmeyen etiket ID'si
                            self._log_unknown_label(label_file.name, line_num + 1, original_class_id)
                            self.stats["lines_skipped_unknown"] += 1
                            # Bu satır yeni listeye eklenmeyecek, yani atlanmış olacak.

                # 2. Temizlenmiş Dosyayı Yaz
                if not self.dry_run and new_lines:
                    output_file_path = output_labels_path / label_file.name
                    with open(output_file_path, 'w', encoding='utf-8') as outfile:
                        outfile.write('\n'.join(new_lines) + '\n')

                    # 3. İlgili Görüntüyü Kopyala
                    image_file_stem = label_file.stem
                    for ext in ['.jpg', '.jpeg', '.png']:
                        source_image = source_images_path / f"{image_file_stem}{ext}"
                        if source_image.is_file():
                            shutil.copy2(source_image, output_images_path)
                            break

                self.stats["txt_files_processed"] += 1
                processed_files_in_subdir += 1

            except Exception as e:
                logging.error(
                    f"    [ERROR][TXT] {self.dataset_name}/{subdir_name}/{label_file.name} işlenirken hata: {e}")
                self.stats["txt_errors"] += 1

        # DEBUG 3: Alt klasör bittiğinde sayaçları yazdır
        logging.debug(
            f"      [DEBUG] Stats for {subdir_name}: Found={self.stats['txt_files_found']}, Processed={self.stats['txt_files_processed']}, Lines={self.stats['lines_processed']}")
        # DEBUG 3 Bitiş



    def process_all_labels(self):
        logging.info(f"    [TXT] {self.dataset_name} için işleme başlıyor...")

        for subdir_name in self.subdirs_to_check:
            self._process_subdir(subdir_name)

        logging.info(f"    [TXT] {self.dataset_name} için işleme tamamlandı.")
        return self.stats

# ----------------------------------------------------------------------------
# BÖLÜM 3: ReportManager Class ve Main Fonksiyonu (Yönetim)
# ----------------------------------------------------------------------------

class ReportManager:
    """ Global sayaçları tutar ve CSV raporunu yönetir. """

    def __init__(self):
        self.global_stats = self._initialize_global_stats()

    def _initialize_global_stats(self):
        return {
            "total_datasets_found": 0, "datasets_ready_count": 0,
            "total_label_files_found": 0, "total_label_files_processed": 0,
            "total_lines_processed": 0, "total_lines_modified": 0,
            "total_unknown_labels_skipped": 0, "total_lines_skipped_invalid_format": 0,
            "total_txt_errors": 0
        }

    def update_global_stats(self, txt_stats: dict):
        """ İşlemciden gelen istatistikleri genel sayaçlara ekler. """
        for key, value in txt_stats.items():
            if key in self.global_stats:
                self.global_stats[key] += value

    def get_final_summary(self):
        return self.global_stats

    def write_report(self, report_path: Path, report_data: list):
        """ Raporu CSV dosyasına yazar. """
        report_header = list(report_data[0].keys())

        # TXT istatistiklerini başlığa ekle
        txt_stats_headers = [
            "txt_files_found", "txt_files_processed", "lines_processed", "lines_modified",
            "skipped_unknown_lines", "skipped_invalid_lines", "txt_errors"
        ]
        report_header.extend(h for h in txt_stats_headers if h not in report_header)

        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=report_header, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(report_data)
            logging.info(f"[INFO] Rapor başarıyla '{report_path}' dosyasına yazıldı.")
        except Exception as e:
            logging.error(f"[HATA] Rapor dosyası yazılamadı: {e}")


def main():
    parser = argparse.ArgumentParser(description="Mega-Set Termal Etiket Temizleyici (OOP).")
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                        help='Ham veri setlerinin ana klasörü.')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Temizlenmiş verilerin yazılacağı klasör.')
    parser.add_argument('--report_file', type=str, default=DEFAULT_REPORT_FILE,
                        help='CSV rapor dosyasının adı.')
    parser.add_argument('--dry_run', action='store_true',
                        help='Sadece tara ve raporla, dosya yazma/değiştirme.')

    args = parser.parse_args()

    base_data_path = Path(args.data_dir)
    output_base_path = Path(args.output_dir)
    report_path = Path(args.report_file)

    if not base_data_path.is_dir():
        logging.error(f"[FATAL] Kaynak klasör bulunamadı: {base_data_path}")
        sys.exit(1)

    ERROR_LOG_FILE = output_base_path / 'cleaning_unknown_labels.log'

    try:
        output_base_path.mkdir(parents=True, exist_ok=True)
        if ERROR_LOG_FILE.is_file():
            ERROR_LOG_FILE.unlink()
    except Exception as e:
        logging.error(f"[FATAL] Çıktı klasörü veya log dosyası hazırlanamadı: {e}");
        sys.exit(1)

    logging.info("--- MEGA-SET TEMİZLEYİCİ BAŞLADI (OOP) ---")
    logging.info(f"Mod: {'DRY RUN (Değişiklik Yapılmayacak)' if args.dry_run else 'UYGULAMA MODU'}")
    logging.info(f"Bilinmeyen Etiketler Logu: {ERROR_LOG_FILE}")

    report_manager = ReportManager()
    report_data = []

    all_dataset_paths = []
    try:
        all_dataset_paths = sorted([p for p in base_data_path.iterdir() if p.is_dir() and not p.name.startswith('.')])
        report_manager.global_stats["total_datasets_found"] = len(all_dataset_paths)
        logging.info(f"Bulunan potansiyel veri seti klasör sayısı: {len(all_dataset_paths)}")
    except Exception as e:
        logging.error(f"[FATAL] Klasör listelenemedi: {e}");
        sys.exit(1)

    # --- ANA İŞLEME DÖNGÜSÜ ---
    for index, dataset_path in enumerate(all_dataset_paths):
        dataset_name = dataset_path.name
        logging.info(f"\n[INFO] İşleniyor ({index + 1}/{len(all_dataset_paths)}): {dataset_name}")

        analyzer = DatasetAnalyzer(dataset_path)
        is_ready = analyzer.validate_and_map()

        report_row = analyzer.get_report_row()
        txt_stats = {}

        if is_ready:
            report_manager.global_stats["datasets_ready_count"] += 1
            output_dataset_path = output_base_path / dataset_name

            try:
                # 1. YAML dosyasını güncelle
                analyzer.update_data_yaml(output_dataset_path, args.dry_run)
                
                # 2. Label dosyalarını işle
                processor = LabelProcessor(dataset_path, output_dataset_path,
                                           analyzer.get_mapping(), args.dry_run, ERROR_LOG_FILE)
                txt_stats = processor.process_all_labels()

                # --- RAPORLAMA DÜZELTMESİ (Grok Haklıydı!) ---
                # txt_stats sadece TXT işleme verilerini içerir. Test Missing bilgisini Analyzer'dan almalıyız.
                report_row["test_missing"] = analyzer.report["test_missing"]  # <<< YENİ EKLENDİ
                # ---------------------------------------------

                report_row.update(txt_stats)
                report_manager.update_global_stats(txt_stats)
            except Exception as e:
                logging.error(f"  [FATAL_TXT] İşleme sırasında beklenmedik hata: {e}")
                report_row["ready_for_txt_processing"] = False
                report_row["txt_errors"] = report_row.get("txt_errors", 0) + 1
        else:
            # İşlenmeye hazır değilse de test_missing bilgisini ekle
            report_row["test_missing"] = analyzer.report["test_missing"]  # <<< YENİ EKLENDİ

        report_data.append(report_row)

    # ... (SONUÇLANDIRMA kısmı aynı) ...

    logging.info("\n--- İŞLEME TAMAMLANDI ---")

    report_manager.write_report(report_path, report_data)

    summary = report_manager.get_final_summary()
    logging.info("\n--- GENEL ÖZET ---")
    logging.info(f"Toplam Klasör Taraması         : {summary['total_datasets_found']}")
    logging.info(f"TXT İşlemeye Hazır Klasör Sayısı: {summary['datasets_ready_count']}")
    logging.info(f"Toplam İşlenen Satır Sayısı    : {summary['total_lines_processed']}")
    logging.info(
        f"Toplam Değiştirilen Satır Sayısı: {summary['total_lines_modified']} {'(Dry Run)' if args.dry_run else ''}")
    logging.info(
        f"Toplam Atlanan Satır Sayısı     : {summary['total_unknown_labels_skipped'] + summary['total_lines_skipped_invalid_format']}")


if __name__ == "__main__":
    main()