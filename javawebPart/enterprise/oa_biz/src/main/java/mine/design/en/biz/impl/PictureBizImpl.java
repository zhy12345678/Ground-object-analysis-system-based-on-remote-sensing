package mine.design.en.biz.impl;

import mine.design.en.biz.PictureBiz;
import mine.design.en.dao.PictureDao;
import mine.design.en.entity.Picture;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
@Service("PictureBiz")
public class PictureBizImpl implements PictureBiz {
    @Autowired
    private PictureDao pictureDao;
    public void add(Picture picture) {
        pictureDao.insert(picture);
    }
    public List<Picture> getTime(){return pictureDao.selectTime();}
    public List<Picture> getAccurate() {
        return pictureDao.selectAccurate();
    }
    public List<Picture> getAll() {
        return pictureDao.getAll();
    }

}
