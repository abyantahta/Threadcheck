"""Aksi profil kamera (dipakai header dan halaman Pengaturan)."""
from .. import config, storage
from ..config import CLASS_NG, CLASS_OK
from ..qt import QtWidgets


def _busy(parent, ctx) -> bool:
    if getattr(ctx, "training_active", False):
        QtWidgets.QMessageBox.information(parent, "Profil kamera",
                                          "Tunggu training selesai sebelum mengganti atau mengubah profil.")
        return True
    return False


def profile_summary(name: str) -> str:
    n_ok = len(storage.list_samples(CLASS_OK, name))
    n_ng = len(storage.list_samples(CLASS_NG, name))
    has_model = config.model_path(name).exists()
    return f"{n_ok} ulir · {n_ng} tanpa ulir · {'model ada' if has_model else 'belum ada model'}"


def switch(parent, ctx, name: str) -> bool:
    if name == ctx.profile:
        return True
    if _busy(parent, ctx):
        return False
    ctx.switch_profile(name)
    return True


def create(parent, ctx):
    if _busy(parent, ctx):
        return
    name, ok = QtWidgets.QInputDialog.getText(
        parent, "Profil kamera baru",
        "Nama profil (mis. Endoskopi 1080p):")
    if not ok:
        return
    err = config.profile_name_error(name)
    if err:
        QtWidgets.QMessageBox.warning(parent, "Profil kamera baru", err)
        return
    r = QtWidgets.QMessageBox.question(
        parent, "Profil kamera baru",
        f"Salin pengaturan dari profil '{ctx.profile}' (kamera, ROI, analisis, ambang)?\n\n"
        "Pilih No untuk memulai dari pengaturan bawaan.\n"
        "Dataset dan model TIDAK disalin: profil baru perlu sampel & training sendiri.")
    ctx.create_profile(name.strip(), copy_current=(r == QtWidgets.QMessageBox.Yes))


def rename(parent, ctx):
    if _busy(parent, ctx):
        return
    old = ctx.profile
    name, ok = QtWidgets.QInputDialog.getText(parent, "Ganti nama profil", "Nama baru:", text=old)
    if not ok or name.strip() == old:
        return
    err = config.profile_name_error(name, exclude=old)
    if err:
        QtWidgets.QMessageBox.warning(parent, "Ganti nama profil", err)
        return
    ctx.rename_profile(old, name.strip())


def delete(parent, ctx, name: str):
    if _busy(parent, ctx):
        return
    if name == ctx.profile:
        QtWidgets.QMessageBox.information(parent, "Hapus profil",
                                          "Profil yang sedang aktif tidak bisa dihapus. Aktifkan profil lain dulu.")
        return
    r = QtWidgets.QMessageBox.warning(
        parent, "Hapus profil",
        f"Hapus profil '{name}'?\n\n{profile_summary(name)}\n\n"
        "Dataset, model, dan pengaturan profil ini akan dihapus permanen.\n"
        "Gambar hasil inspeksi dan log tetap disimpan.",
        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
    if r == QtWidgets.QMessageBox.Yes:
        ctx.delete_profile(name)
